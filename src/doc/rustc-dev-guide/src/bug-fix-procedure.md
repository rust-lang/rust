# Procedures for breaking changes

This page defines the best practices procedure for making bug fixes or soundness
corrections in the compiler that can cause existing code to stop compiling. This
text is based on
[RFC 1589](https://github.com/rust-lang/rfcs/blob/master/text/1589-rustc-bug-fix-procedure.md).

# Motivation

[motivation]: #motivation

From time to time, we encounter the need to make a bug fix, soundness
correction, or other change in the compiler which will cause existing code to
stop compiling. When this happens, it is important that we handle the change in
a way that gives users of Rust a smooth transition. What we want to avoid is
that existing programs suddenly stop compiling with opaque error messages: we
would prefer to have a gradual period of warnings, with clear guidance as to
what the problem is, how to fix it, and why the change was made. This RFC
describes the procedure that we have been developing for handling breaking
changes that aims to achieve that kind of smooth transition.

One of the key points of this policy is that (a) warnings should be issued
initially rather than hard errors if at all possible and (b) every change that
causes existing code to stop compiling will have an associated tracking issue.
This issue provides a point to collect feedback on the results of that change.
Sometimes changes have unexpectedly large consequences or there may be a way to
avoid the change that was not considered. In those cases, we may decide to
change course and roll back the change, or find another solution (if warnings
are being used, this is particularly easy to do).

### What qualifies as a bug fix?

Note that this RFC does not try to define when a breaking change is permitted.
That is already covered under [RFC 1122][]. This document assumes that the
change being made is in accordance with those policies. Here is a summary of the
conditions from RFC 1122:

- **Soundness changes:** Fixes to holes uncovered in the type system.
- **Compiler bugs:** Places where the compiler is not implementing the specified
  semantics found in an RFC or lang-team decision.
- **Underspecified language semantics:** Clarifications to grey areas where the
  compiler behaves inconsistently and no formal behavior had been previously
  decided.

Please see [the RFC][rfc 1122] for full details!

# Detailed design

[design]: #detailed-design

The procedure for making a breaking change is as follows (each of these steps is
described in more detail below):

1. Do a **crater run** to assess the impact of the change.
2. Make a **special tracking issue** dedicated to the change.
3. Do not report an error right away. Instead, **issue forwards-compatibility
   lint warnings**.
   - Sometimes this is not straightforward. See the text below for suggestions
     on different techniques we have employed in the past.
   - For cases where warnings are infeasible:
     - Report errors, but make every effort to give a targeted error message
       that directs users to the tracking issue
     - Submit PRs to all known affected crates that fix the issue
       - or, at minimum, alert the owners of those crates to the problem and
         direct them to the tracking issue
4. Once the change has been in the wild for at least one cycle, we can
   **stabilize the change**, converting those warnings into errors.

Finally, for changes to `rustc_ast` that will affect plugins, the general policy
is to batch these changes. That is discussed below in more detail.

### Tracking issue

Every breaking change should be accompanied by a **dedicated tracking issue**
for that change. The main text of this issue should describe the change being
made, with a focus on what users must do to fix their code. The issue should be
approachable and practical; it may make sense to direct users to an RFC or some
other issue for the full details. The issue also serves as a place where users
can comment with questions or other concerns.

A template for these breaking-change tracking issues can be found
[here][template]. An example of how such an issue should look can be [found
here][breaking-change-issue].

[template]: https://github.com/rust-lang/rust/issues/new?template=tracking_issue_future.md

### Issuing future compatibility warnings

The best way to handle a breaking change is to begin by issuing
future-compatibility warnings. These are a special category of lint warning.
Adding a new future-compatibility warning can be done as follows.

```rust
// 1. Define the lint in `compiler/rustc_middle/src/lint/builtin.rs`:
declare_lint! {
    pub YOUR_ERROR_HERE,
    Warn,
    "illegal use of foo bar baz"
}

// 2. Add to the list of HardwiredLints in the same file:
impl LintPass for HardwiredLints {
    fn get_lints(&self) -> LintArray {
        lint_array!(
            ..,
            YOUR_ERROR_HERE
        )
    }
}

// 3. Register the lint in `compiler/rustc_lint/src/lib.rs`:
store.register_future_incompatible(sess, vec![
    ...,
    FutureIncompatibleInfo {
        id: LintId::of(YOUR_ERROR_HERE),
        reference: "issue #1234", // your tracking issue here!
    },
]);

// 4. Report the lint:
tcx.lint_node(
    lint::builtin::YOUR_ERROR_HERE,
    path_id,
    binding.span,
    format!("some helper message here"));
```

#### Helpful techniques

It can often be challenging to filter out new warnings from older, pre-existing
errors. One technique that has been used in the past is to run the older code
unchanged and collect the errors it would have reported. You can then issue
warnings for any errors you would give which do not appear in that original set.
Another option is to abort compilation after the original code completes if
errors are reported: then you know that your new code will only execute when
there were no errors before.

#### Crater and crates.io

[Crater] is a bot that will compile all crates.io crates and many
public github repos with the compiler with your changes. A report will then be
generated with crates that ceased to compile with or began to compile with your
changes. Crater runs can take a few days to complete.

[Crater]: ./tests/crater.md

We should always do a crater run to assess impact. It is polite and considerate
to at least notify the authors of affected crates the breaking change. If we can
submit PRs to fix the problem, so much the better.

#### Is it ever acceptable to go directly to issuing errors?

Changes that are believed to have negligible impact can go directly to issuing
an error. One rule of thumb would be to check against `crates.io`: if fewer than
10 **total** affected projects are found (**not** root errors), we can move
straight to an error. In such cases, we should still make the "breaking change"
page as before, and we should ensure that the error directs users to this page.
In other words, everything should be the same except that users are getting an
error, and not a warning. Moreover, we should submit PRs to the affected
projects (ideally before the PR implementing the change lands in rustc).

If the impact is not believed to be negligible (e.g., more than 10 crates are
affected), then warnings are required (unless the compiler team agrees to grant
a special exemption in some particular case). If implementing warnings is not
feasible, then we should make an aggressive strategy of migrating crates before
we land the change so as to lower the number of affected crates. Here are some
techniques for approaching this scenario:

1. Issue warnings for subparts of the problem, and reserve the new errors for
   the smallest set of cases you can.
2. Try to give a very precise error message that suggests how to fix the problem
   and directs users to the tracking issue.
3. It may also make sense to layer the fix:
   - First, add warnings where possible and let those land before proceeding to
     issue errors.
   - Work with authors of affected crates to ensure that corrected versions are
     available _before_ the fix lands, so that downstream users can use them.

### Stabilization

After a change is made, we will **stabilize** the change using the same process
that we use for unstable features:

- After a new release is made, we will go through the outstanding tracking
  issues corresponding to breaking changes and nominate some of them for **final
  comment period** (FCP).
- The FCP for such issues lasts for one cycle. In the final week or two of the
  cycle, we will review comments and make a final determination:

  - Convert to error: the change should be made into a hard error.
  - Revert: we should remove the warning and continue to allow the older code to
    compile.
  - Defer: can't decide yet, wait longer, or try other strategies.

Ideally, breaking changes should have landed on the **stable branch** of the
compiler before they are finalized.

<a id="guide"></a>

### Removing a lint

Once we have decided to make a "future warning" into a hard error, we need a PR
that removes the custom lint. As an example, here are the steps required to
remove the `overlapping_inherent_impls` compatibility lint. First, convert the
name of the lint to uppercase (`OVERLAPPING_INHERENT_IMPLS`) ripgrep through the
source for that string. We will basically by converting each place where this
lint name is mentioned (in the compiler, we use the upper-case name, and a macro
automatically generates the lower-case string; so searching for
`overlapping_inherent_impls` would not find much).

> NOTE: these exact files don't exist anymore, but the procedure is still the same.

#### Remove the lint.

The first reference you will likely find is the lint definition [in
`rustc_session/src/lint/builtin.rs` that resembles this][defsource]:

[defsource]: https://github.com/rust-lang/rust/blob/085d71c3efe453863739c1fb68fd9bd1beff214f/src/librustc/lint/builtin.rs#L171-L175

```rust
declare_lint! {
    pub OVERLAPPING_INHERENT_IMPLS,
    Deny, // this may also say Warning
    "two overlapping inherent impls define an item with the same name were erroneously allowed"
}
```

This `declare_lint!` macro creates the relevant data structures. Remove it. You
will also find that there is a mention of `OVERLAPPING_INHERENT_IMPLS` later in
the file as [part of a `lint_array!`][lintarraysource]; remove it too.

[lintarraysource]: https://github.com/rust-lang/rust/blob/085d71c3efe453863739c1fb68fd9bd1beff214f/src/librustc/lint/builtin.rs#L252-L290

Next, you see [a reference to `OVERLAPPING_INHERENT_IMPLS` in
`rustc_lint/src/lib.rs`][futuresource]. This is defining the lint as a "future
compatibility lint":

```rust
FutureIncompatibleInfo {
    id: LintId::of(OVERLAPPING_INHERENT_IMPLS),
    reference: "issue #36889 <https://github.com/rust-lang/rust/issues/36889>",
},
```

Remove this too.

#### Add the lint to the list of removed lints.

In `compiler/rustc_lint/src/lib.rs` there is a list of "renamed and removed lints".
You can add this lint to the list:

```rust
store.register_removed("overlapping_inherent_impls", "converted into hard error, see #36889");
```

where `#36889` is the tracking issue for your lint.

#### Update the places that issue the lint

Finally, the last class of references you will see are the places that actually
**trigger** the lint itself (i.e., what causes the warnings to appear). These
you do not want to delete. Instead, you want to convert them into errors. In
this case, the [`add_lint` call][addlintsource] looks like this:

```rust
self.tcx.sess.add_lint(lint::builtin::OVERLAPPING_INHERENT_IMPLS,
                       node_id,
                       self.tcx.span_of_impl(item1).unwrap(),
                       msg);
```

We want to convert this into an error. In some cases, there may be an
existing error for this scenario. In others, we will need to allocate a
fresh diagnostic code.  [Instructions for allocating a fresh diagnostic
code can be found here.](./diagnostics/error-codes.md) You may want
to mention in the extended description that the compiler behavior
changed on this point, and include a reference to the tracking issue for
the change.

Let's say that we've adopted `E0592` as our code. Then we can change the
`add_lint()` call above to something like:

```rust
struct_span_code_err!(self.dcx(), self.tcx.span_of_impl(item1).unwrap(), E0592, msg)
    .emit();
```

#### Update tests

Finally, run the test suite. These should be some tests that used to reference
the `overlapping_inherent_impls` lint, those will need to be updated. In
general, if the test used to have `#[deny(overlapping_inherent_impls)]`, that
can just be removed.

```
./x test
```

#### All done!

Open a PR. =)

[addlintsource]: https://github.com/rust-lang/rust/blob/085d71c3efe453863739c1fb68fd9bd1beff214f/src/librustc_typeck/coherence/inherent.rs#L300-L303
[futuresource]: https://github.com/rust-lang/rust/blob/085d71c3efe453863739c1fb68fd9bd1beff214f/src/librustc_lint/lib.rs#L202-L205

<!-- -Links--------------------------------------------------------------------- -->

[rfc 1122]: https://github.com/rust-lang/rfcs/blob/master/text/1122-language-semver.md
[breaking-change-issue]: https://gist.github.com/nikomatsakis/631ec8b4af9a18b5d062d9d9b7d3d967
