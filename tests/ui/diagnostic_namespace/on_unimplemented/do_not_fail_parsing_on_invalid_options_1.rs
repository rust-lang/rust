//@ reference: attributes.diagnostic.on_unimplemented.syntax
//@ reference: attributes.diagnostic.on_unimplemented.unknown-keys
#[diagnostic::on_unimplemented(unsupported = "foo")]
//~^WARN malformed `on_unimplemented` attribute
//~|WARN malformed `on_unimplemented` attribute
trait Foo {}

#[diagnostic::on_unimplemented(message = "Baz")]
//~^WARN `#[diagnostic::on_unimplemented]` can only be applied to trait definitions
struct Bar {}

#[diagnostic::on_unimplemented(message = "Boom", unsupported = "Bar")]
//~^WARN malformed `on_unimplemented` attribute
//~|WARN malformed `on_unimplemented` attribute
trait Baz {}

#[diagnostic::on_unimplemented(message = "Boom", on(Self = "i32", message = "whatever"))]
//~^WARN malformed `on_unimplemented` attribute
//~|WARN malformed `on_unimplemented` attribute
trait Boom {}

#[diagnostic::on_unimplemented(message = "Boom", on(_Self = "i32", message = "whatever"))]
//~^WARN malformed `on_unimplemented` attribute
trait _Self {}

#[diagnostic::on_unimplemented = "boom"]
//~^WARN malformed `on_unimplemented` attribute
trait Doom {}

#[diagnostic::on_unimplemented]
//~^WARN missing options for `on_unimplemented` attribute
//~|WARN missing options for `on_unimplemented` attribute
trait Whatever {}

#[diagnostic::on_unimplemented(message = "{DoesNotExist}")]
//~^WARN there is no parameter `DoesNotExist` on trait `Test`
//~|WARN there is no parameter `DoesNotExist` on trait `Test`
trait Test {}

fn take_foo(_: impl Foo) {}
fn take_baz(_: impl Baz) {}
fn take_boom(_: impl Boom) {}
fn take_whatever(_: impl Whatever) {}
fn take_test(_: impl Test) {}

fn main() {
    take_foo(1_i32);
    //~^ERROR the trait bound `i32: Foo` is not satisfied
    take_baz(1_i32);
    //~^ERROR Boom
    take_boom(1_i32);
    //~^ERROR Boom
    take_whatever(1_i32);
    //~^ERROR the trait bound `i32: Whatever` is not satisfied
    take_test(());
    //~^ERROR {DoesNotExist}
}
