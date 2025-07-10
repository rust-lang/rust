//@ reference: attributes.diagnostic.on_unimplemented.invalid-string
#[diagnostic::on_unimplemented(message = "{{Test } thing")]
//~^WARN unmatched `}` found
//~|WARN unmatched `}` found
trait ImportantTrait1 {}

#[diagnostic::on_unimplemented(message = "Test {}")]
//~^WARN positional format arguments are not allowed here
//~|WARN positional format arguments are not allowed here
trait ImportantTrait2 {}

#[diagnostic::on_unimplemented(message = "Test {1:}")]
//~^WARN positional format arguments are not allowed here
//~|WARN positional format arguments are not allowed here
//~|WARN invalid format specifier [malformed_diagnostic_format_literals]
//~|WARN invalid format specifier [malformed_diagnostic_format_literals]
trait ImportantTrait3 {}

#[diagnostic::on_unimplemented(message = "Test {Self:123}")]
//~^WARN invalid format specifier
//~|WARN invalid format specifier
trait ImportantTrait4 {}

#[diagnostic::on_unimplemented(message = "Test {Self:!}")]
//~^WARN invalid format specifier [malformed_diagnostic_format_literals]
//~|WARN invalid format specifier [malformed_diagnostic_format_literals]
trait ImportantTrait5 {}

#[diagnostic::on_unimplemented(message = "Test {Self:}")]
//~^WARN invalid format specifier [malformed_diagnostic_format_literals]
//~|WARN invalid format specifier [malformed_diagnostic_format_literals]
trait ImportantTrait6 {}


fn check_1(_: impl ImportantTrait1) {}
fn check_2(_: impl ImportantTrait2) {}
fn check_3(_: impl ImportantTrait3) {}
fn check_4(_: impl ImportantTrait4) {}
fn check_5(_: impl ImportantTrait5) {}
fn check_6(_: impl ImportantTrait6) {}

fn main() {
    check_1(());
    //~^ERROR {{Test } thing
    check_2(());
    //~^ERROR Test {}
    check_3(());
    //~^ERROR Test {1}
    check_4(());
    //~^ERROR Test ()
    check_5(());
    //~^ERROR Test ()
    check_6(());
    //~^ERROR Test ()
}
