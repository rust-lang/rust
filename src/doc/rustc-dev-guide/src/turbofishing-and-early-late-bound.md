# Turbofishing's interactions with early/late bound parameters

The early/late bound parameter distinction on functions introduces some complications when providing generic arguments to functions. This document discusses what those are and how they might interact with future changes to make more things late bound.

## Can't turbofish generic arguments on functions sometimes

When a function has any late bound lifetime parameters (be they explicitly defined or implicitly introduced via lifetime elision) we disallow specifying any lifetime arguments on the function. Sometimes this is a hard error other times it is a future compat lint ([`late_bound_lifetime_arguments`](https://github.com/rust-lang/rust/issues/42868)).

```rust
fn early<'a: 'a>(a: &'a ()) -> &'a () { a }
fn late<'a>(a: &'a ()) -> &'a () { a }

fn mixed<'a, 'b: 'b>(a: &'a (), b: &'b ()) -> &'a () { a }

struct Foo;
impl Foo {
    fn late<'a>(self, a: &'a ()) -> &'a () { a }
}

fn main() {
    // fine
    let f = early::<'static>;
    
    // some variation of hard errors and future compat lints
    Foo.late::<'static>(&());
    let f = late::<'static>;
    let f = mixed::<'static, 'static>;
    let f = mixed::<'static>;
    late::<'static>(&());
}
```

The justification for this is that late bound parameters are not present on the `FnDef` so the arguments to late bound parameters can't be present in the substs for the type. i.e. the `late` function in the above code snippet would not have any generic parameters on the `FnDef` zst:
```rust
// example desugaring of the `late` function and its zst + builtin Fn impl
struct LateFnDef;
impl<'a> Fn<(&'a ())> for LateFnDef {
    type Output = &'a ();
    ...
}
```

The cause for some situations giving future compat lints and others giving hard errors is a little arbitrary but explainable:
- It's always a hard error for method calls
- It's only a hard error on paths to free functions if there is no unambiguous way to create the substs for the fndef from the lifetime arguments. (i.e. the amount of lifetimes provided must be exactly equal to the amount of early bound lifetimes or else it's a hard error)

## Back compat issues from turning early bound to late bound

Because of the previously mentioned restriction on turbofishing generic arguments, it is a breaking change to upgrade a lifetime from early bound to late bound as it can cause existing turbofishies to become hard errors/future compat lints.

Many t-types members have expressed interest in wanting more parameters to be late bound. We cannot do so if making something late bound is going to break code that many would expect to work (judging by the future compat lint issue many people do expect to be able to turbofish late bound parameters).

## Interactions with late bound type/const parameters

If we were to make some type/const parameters late bound we would definitely not want to disallow turbofishing them as it presumably(?) would break a Tonne of code. 

While lifetimes do differ from type/consts in some ways I(BoxyUwU) do not believe there is any justification for why it would make sense to allow turbofishing late bound type/const parameters but not late bound lifetimes.

## Removing the hard error/fcw

From reasons above it seems reasonable that we may want to remove the hard error and fcw (removing the errors/fcw is definitely a blocker for making more things late bound).

example behaviour:
```rust
fn late<'a>(a: &'a ()) -> &'a () { a }

fn accepts_fn(_: impl for<'a> Fn(&'a ()) -> &'a ()) {}
fn accepts_fn_2(_: impl Fn(&'static ()) -> &'static ()) {}

fn main() {
    let f = late::<'static>;
    
    accepts_fn(f); //~ error: `f` doesnt implement `for<'a> Fn(&'a ()) -> &'a ()`
    accepts_fn_2(f) // works
    
    accepts_fn(late) // works
}
````

one potential complication is that we would want a way to specify a generic argument to a function without having to specify arguments for all previous parameters. i.e. ideally you could write the following code somehow.
```rust
fn late<'a, 'b>(_: &'a (), _: &'b ()) {}

fn accepts_fn(_: impl for<'a> Fn(&'a (), &'static ())) {}

fn main() {
    // a naive implementation would have a `ReInfer` as the subst for `'a` parameter
    // no longer allowing the FnDef to satisfy the `for<'a> Fn(&'a ()` bound
    let f = late::<'_, 'static>;
    accepts_fn(f);
}
```
Maybe we can just special case astconv for `_`/`'_` arguments for late bound parameters somehow and have it not mean the same thing as `_` for early bound parameters. Regardless I think we would need a solution that would allow writing the above code even if it was done by some new syntax such as havign to write `late::<k#no_argument, 'static>` (naturally `k#no_argument` would only make sense as an argument to late bound parameters).


## Conclusion

Late bound params make turbofishing complicated. We currently have a hard error and a future compat lint that we might or might not want.

We don't have to decide on anything in this meeting when it comes to the error/fcw or whether we even want to make more things late bound. The primary purpose is to spread knowledge. I would still like to see what people think about removing the error and fcw though (but any decision about it will have a proper FCP).
