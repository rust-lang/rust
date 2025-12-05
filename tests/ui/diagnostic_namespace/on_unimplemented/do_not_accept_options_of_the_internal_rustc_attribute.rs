//@ reference: attributes.diagnostic.on_unimplemented.format-parameters
//@ reference: attributes.diagnostic.on_unimplemented.keys
//@ reference: attributes.diagnostic.on_unimplemented.syntax
//@ reference: attributes.diagnostic.on_unimplemented.invalid-formats
#[diagnostic::on_unimplemented(
    on(Self = "&str"),
    //~^WARN malformed `on_unimplemented` attribute
    //~|WARN malformed `on_unimplemented` attribute
    message = "trait has `{Self}` and `{T}` as params",
    label = "trait has `{Self}` and `{T}` as params",
    note  = "trait has `{Self}` and `{T}` as params",
    parent_label = "in this scope",
    //~^WARN malformed `on_unimplemented` attribute
    //~|WARN malformed `on_unimplemented` attribute
    append_const_msg
    //~^WARN malformed `on_unimplemented` attribute
    //~|WARN malformed `on_unimplemented` attribute
)]
trait Foo<T> {}

#[diagnostic::on_unimplemented = "Message"]
//~^WARN malformed `on_unimplemented` attribute
//~|WARN malformed `on_unimplemented` attribute
trait Bar {}

#[diagnostic::on_unimplemented(message = "Not allowed to apply it on a impl")]
//~^WARN #[diagnostic::on_unimplemented]` can only be applied to trait definitions
impl Bar for i32 {}

// cannot use special rustc_on_unimplement symbols
// in the format string
#[diagnostic::on_unimplemented(
    message = "{from_desugaring}{direct}{cause}{integral}{integer}",
    //~^WARN there is no parameter `from_desugaring` on trait `Baz`
    //~|WARN there is no parameter `from_desugaring` on trait `Baz`
    //~|WARN there is no parameter `direct` on trait `Baz`
    //~|WARN there is no parameter `direct` on trait `Baz`
    //~|WARN there is no parameter `cause` on trait `Baz`
    //~|WARN there is no parameter `cause` on trait `Baz`
    //~|WARN there is no parameter `integral` on trait `Baz`
    //~|WARN there is no parameter `integral` on trait `Baz`
    //~|WARN there is no parameter `integer` on trait `Baz`
    //~|WARN there is no parameter `integer` on trait `Baz`
    label = "{float}{_Self}{crate_local}{Trait}{ItemContext}{This}"
    //~^WARN there is no parameter `float` on trait `Baz`
    //~|WARN there is no parameter `float` on trait `Baz`
    //~|WARN there is no parameter `_Self` on trait `Baz`
    //~|WARN there is no parameter `_Self` on trait `Baz`
    //~|WARN there is no parameter `crate_local` on trait `Baz`
    //~|WARN there is no parameter `crate_local` on trait `Baz`
    //~|WARN there is no parameter `Trait` on trait `Baz`
    //~|WARN there is no parameter `Trait` on trait `Baz`
    //~|WARN there is no parameter `ItemContext` on trait `Baz`
    //~|WARN there is no parameter `ItemContext` on trait `Baz`
    //~|WARN there is no parameter `This` on trait `Baz`
    //~|WARN there is no parameter `This` on trait `Baz`
)]
trait Baz {}

fn takes_foo(_: impl Foo<i32>) {}
fn takes_bar(_: impl Bar) {}
fn takes_baz(_: impl Baz) {}

fn main() {
    takes_foo(());
    //~^ERROR trait has `()` and `i32` as params
    takes_bar(());
    //~^ERROR the trait bound `(): Bar` is not satisfied
    takes_baz(());
    //~^ERROR {from_desugaring}{direct}{cause}{integral}{integer}
}
