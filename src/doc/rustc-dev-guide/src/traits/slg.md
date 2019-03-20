# The On-Demand SLG solver

Given a set of program clauses (provided by our [lowering rules][lowering])
and a query, we need to return the result of the query and the value of any
type variables we can determine. This is the job of the solver.

For example, `exists<T> { Vec<T>: FromIterator<u32> }` has one solution, so
its result is `Unique; substitution [?T := u32]`. A solution also comes with
a set of region constraints, which we'll ignore in this introduction.

[lowering]: ./lowering-rules.html

## Goals of the Solver

### On demand

There are often many, or even infinitely many, solutions to a query. For
example, say we want to prove that `exists<T> { Vec<T>: Debug }` for _some_
type `?T`. Our solver should be capable of yielding one answer at a time, say
`?T = u32`, then `?T = i32`, and so on, rather than iterating over every type
in the type system. If we need more answers, we can request more until we are
done. This is similar to how Prolog works.

*See also: [The traditional, interactive Prolog query][pq]*

[pq]: ./canonical-queries.html#the-traditional-interactive-prolog-query

### Breadth-first

`Vec<?T>: Debug` is true if `?T: Debug`. This leads to a cycle: `[Vec<u32>,
Vec<Vec<u32>>, Vec<Vec<Vec<u32>>>]`, and so on all implement `Debug`. Our
solver ought to be breadth first and consider answers like `[Vec<u32>: Debug,
Vec<i32>: Debug, ...]` before it recurses, or we may never find the answer
we're looking for.

### Cachable

To speed up compilation, we need to cache results, including partial results
left over from past solver queries.

## Description of how it works

The basis of the solver is the [`Forest`] type. A *forest* stores a
collection of *tables* as well as a *stack*. Each *table* represents
the stored results of a particular query that is being performed, as
well as the various *strands*, which are basically suspended
computations that may be used to find more answers. Tables are
interdependent: solving one query may require solving others.

[`Forest`]: https://rust-lang.github.io/chalk/doc/chalk_engine/forest/struct.Forest.html

### Walkthrough

Perhaps the easiest way to explain how the solver works is to walk
through an example. Let's imagine that we have the following program:

```rust,ignore
trait Debug { }

struct u32 { }
impl Debug for u32 { }

struct Rc<T> { }
impl<T: Debug> Debug for Rc<T> { }

struct Vec<T> { }
impl<T: Debug> Debug for Vec<T> { }
```

Now imagine that we want to find answers for the query `exists<T> { Rc<T>:
Debug }`. The first step would be to u-canonicalize this query; this is the
act of giving canonical names to all the unbound inference variables based on
the order of their left-most appearance, as well as canonicalizing the
universes of any universally bound names (e.g., the `T` in `forall<T> { ...
}`). In this case, there are no universally bound names, but the canonical
form Q of the query might look something like:

```text
Rc<?0>: Debug
```

where `?0` is a variable in the root universe U0. We would then go and
look for a table with this canonical query as the key: since the forest is
empty, this lookup will fail, and we will create a new table T0,
corresponding to the u-canonical goal Q.

**Ignoring negative reasoning and regions.** To start, we'll ignore
the possibility of negative goals like `not { Foo }`. We'll phase them
in later, as they bring several complications.

**Creating a table.** When we first create a table, we also initialize
it with a set of *initial strands*. A "strand" is kind of like a
"thread" for the solver: it contains a particular way to produce an
answer. The initial set of strands for a goal like `Rc<?0>: Debug`
(i.e., a "domain goal") is determined by looking for *clauses* in the
environment. In Rust, these clauses derive from impls, but also from
where-clauses that are in scope. In the case of our example, there
would be three clauses, each coming from the program. Using a
Prolog-like notation, these look like:

```text
(u32: Debug).
(Rc<T>: Debug) :- (T: Debug).
(Vec<T>: Debug) :- (T: Debug).
```

To create our initial strands, then, we will try to apply each of
these clauses to our goal of `Rc<?0>: Debug`. The first and third
clauses are inapplicable because `u32` and `Vec<?0>` cannot be unified
with `Rc<?0>`. The second clause, however, will work.

**What is a strand?** Let's talk a bit more about what a strand *is*. In the code, a strand
is the combination of an inference table, an _X-clause_, and (possibly)
a selected subgoal from that X-clause. But what is an X-clause
([`ExClause`], in the code)? An X-clause pulls together a few things:

- The current state of the goal we are trying to prove;
- A set of subgoals that have yet to be proven;
- There are also a few things we're ignoring for now:
  - delayed literals, region constraints

The general form of an X-clause is written much like a Prolog clause,
but with somewhat different semantics. Since we're ignoring delayed
literals and region constraints, an X-clause just looks like this:

```text
G :- L
```
    
where G is a goal and L is a set of subgoals that must be proven.
(The L stands for *literal* -- when we address negative reasoning, a
literal will be either a positive or negative subgoal.) The idea is
that if we are able to prove L then the goal G can be considered true.

In the case of our example, we would wind up creating one strand, with
an X-clause like so:

```text
(Rc<?T>: Debug) :- (?T: Debug)
```

Here, the `?T` refers to one of the inference variables created in the
inference table that accompanies the strand. (I'll use named variables
to refer to inference variables, and numbered variables like `?0` to
refer to variables in a canonicalized goal; in the code, however, they
are both represented with an index.)

For each strand, we also optionally store a *selected subgoal*. This
is the subgoal after the turnstile (`:-`) that we are currently trying
to prove in this strand. Initally, when a strand is first created,
there is no selected subgoal.

[`ExClause`]: https://rust-lang.github.io/chalk/doc/chalk_engine/struct.ExClause.html

**Activating a strand.** Now that we have created the table T0 and
initialized it with strands, we have to actually try and produce an answer.
We do this by invoking the [`ensure_root_answer`] operation on the table:
specifically, we say `ensure_root_answer(T0, A0)`, meaning "ensure that there
is a 0th answer A0 to query T0".

Remember that tables store not only strands, but also a vector of cached
answers. The first thing that [`ensure_root_answer`] does is to check whether
answer A0 is in this vector. If so, we can just return immediately. In this
case, the vector will be empty, and hence that does not apply (this becomes
important for cyclic checks later on).

When there is no cached answer, [`ensure_root_answer`] will try to produce one.
It does this by selecting a strand from the set of active strands -- the
strands are stored in a `VecDeque` and hence processed in a round-robin
fashion. Right now, we have only one strand, storing the following X-clause
with no selected subgoal:

```text
(Rc<?T>: Debug) :- (?T: Debug)
```

When we activate the strand, we see that we have no selected subgoal,
and so we first pick one of the subgoals to process. Here, there is only
one (`?T: Debug`), so that becomes the selected subgoal, changing
the state of the strand to:

```text
(Rc<?T>: Debug) :- selected(?T: Debug, A0)
```
    
Here, we write `selected(L, An)` to indicate that (a) the literal `L`
is the selected subgoal and (b) which answer `An` we are looking for. We
start out looking for `A0`.

[`ensure_root_answer`]:  https://rust-lang.github.io/chalk/doc/chalk_engine/forest/struct.Forest.html#method.ensure_root_answer

**Processing the selected subgoal.** Next, we have to try and find an
answer to this selected goal. To do that, we will u-canonicalize it
and try to find an associated table. In this case, the u-canonical
form of the subgoal is `?0: Debug`: we don't have a table yet for
that, so we can create a new one, T1. As before, we'll initialize T1
with strands. In this case, there will be three strands, because all
the program clauses are potentially applicable. Those three strands
will be:

- `(u32: Debug) :-`, derived from the program clause `(u32: Debug).`.
  - Note: This strand has no subgoals.
- `(Vec<?U>: Debug) :- (?U: Debug)`, derived from the `Vec` impl.
- `(Rc<?U>: Debug) :- (?U: Debug)`, derived from the `Rc` impl.

We can thus summarize the state of the whole forest at this point as
follows:

```text
Table T0 [Rc<?0>: Debug]
  Strands:
    (Rc<?T>: Debug) :- selected(?T: Debug, A0)
  
Table T1 [?0: Debug]
  Strands:
    (u32: Debug) :-
    (Vec<?U>: Debug) :- (?U: Debug)
    (Rc<?V>: Debug) :- (?V: Debug)
```
    
**Delegation between tables.** Now that the active strand from T0 has
created the table T1, it can try to extract an answer. It does this
via that same `ensure_answer` operation we saw before. In this case,
the strand would invoke `ensure_answer(T1, A0)`, since we will start
with the first answer. This will cause T1 to activate its first
strand, `u32: Debug :-`.

This strand is somewhat special: it has no subgoals at all. This means
that the goal is proven. We can therefore add `u32: Debug` to the set
of *answers* for our table, calling it answer A0 (it is the first
answer). The strand is then removed from the list of strands.

The state of table T1 is therefore:

```text
Table T1 [?0: Debug]
  Answers:
    A0 = [?0 = u32]
  Strand:
    (Vec<?U>: Debug) :- (?U: Debug)
    (Rc<?V>: Debug) :- (?V: Debug)
```

Note that I am writing out the answer A0 as a substitution that can be
applied to the table goal; actually, in the code, the goals for each
X-clause are also represented as substitutions, but in this exposition
I've chosen to write them as full goals, following [NFTD].

[NFTD]: ./bibliography.html#slg

Since we now have an answer, `ensure_answer(T1, A0)` will return `Ok`
to the table T0, indicating that answer A0 is available. T0 now has
the job of incorporating that result into its active strand. It does
this in two ways. First, it creates a new strand that is looking for
the next possible answer of T1. Next, it incorpoates the answer from
A0 and removes the subgoal. The resulting state of table T0 is:

```text
Table T0 [Rc<?0>: Debug]
  Strands:
    (Rc<?T>: Debug) :- selected(?T: Debug, A1)
    (Rc<u32>: Debug) :-
```

We then immediately activate the strand that incorporated the answer
(the `Rc<u32>: Debug` one). In this case, that strand has no further
subgoals, so it becomes an answer to the table T0. This answer can
then be returned up to our caller, and the whole forest goes quiescent
at this point (remember, we only do enough work to generate *one*
answer). The ending state of the forest at this point will be:

```text
Table T0 [Rc<?0>: Debug]
  Answer:
    A0 = [?0 = u32]
  Strands:
    (Rc<?T>: Debug) :- selected(?T: Debug, A1)

Table T1 [?0: Debug]
  Answers:
    A0 = [?0 = u32]
  Strand:
    (Vec<?U>: Debug) :- (?U: Debug)
    (Rc<?V>: Debug) :- (?V: Debug)
```

Here you can see how the forest captures both the answers we have
created thus far *and* the strands that will let us try to produce
more answers later on.

## See also

- [chalk_solve README][readme], which contains links to papers used and
  acronyms referenced in the code
- This section is a lightly adapted version of the blog post [An on-demand
  SLG solver for chalk][slg-blog]
- [Negative Reasoning in Chalk][negative-reasoning-blog] explains the need
  for negative reasoning, but not how the SLG solver does it

[readme]: https://github.com/rust-lang/chalk/blob/239e4ae4e69b2785b5f99e0f2b41fc16b0b4e65e/chalk-engine/src/README.md
[slg-blog]: http://smallcultfollowing.com/babysteps/blog/2018/01/31/an-on-demand-slg-solver-for-chalk/
[negative-reasoning-blog]: http://aturon.github.io/blog/2017/04/24/negative-chalk/
