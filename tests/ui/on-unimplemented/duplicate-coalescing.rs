// Check how separate `on_unimplemented` attributes combine into one directive.
//
// Filtered directives keep source order and run before the combined root directive. Therefore,
// the first matching filter provides each scalar option, while root options act as fallbacks.
// Notes from every matching filter and every root directive remain in source order.

#![feature(rustc_attrs)]
#![allow(internal_features)]

#[rustc_on_unimplemented(
    on(Self = "i32", message = "i32 filter message", note = "first filter note"),
    message = "fallback message",
    //~^ NOTE `message` is first declared here
    note = "first root note"
)]
#[rustc_on_unimplemented(
    on(
        any(Self = "i32", Self = "u32"),
        label = "later filter label",
        note = "second filter note"
    ),
    message = "ignored fallback message",
    //~^ WARN `message` is ignored due to previous definition of `message`
    //~| NOTE `message` is later redundantly declared here
    //~| NOTE `#[warn(malformed_diagnostic_attributes)]`
    note = "second root note"
)]
trait Coalesced {
    type Output;
}

#[diagnostic::on_unimplemented(message = "stable message")]
#[rustc_on_unimplemented(label = "internal label")]
trait StableFirst {
    type Output;
}

#[rustc_on_unimplemented(
    on(Self = "i32", message = "internal filter message"),
    message = "internal message"
)]
#[diagnostic::on_unimplemented(label = "stable label")]
trait InternalFirst {
    type Output;
}

fn main() {
    let _: <i32 as Coalesced>::Output;
    //~^ ERROR i32 filter message
    //~| NOTE later filter label
    //~| NOTE first filter note
    //~| NOTE second filter note
    //~| NOTE first root note
    //~| NOTE second root note

    let _: <u32 as Coalesced>::Output;
    //~^ ERROR fallback message
    //~| NOTE later filter label
    //~| NOTE second filter note
    //~| NOTE first root note
    //~| NOTE second root note

    let _: <() as StableFirst>::Output;
    //~^ ERROR stable message
    //~| NOTE internal label

    let _: <i32 as InternalFirst>::Output;
    //~^ ERROR internal filter message
    //~| NOTE stable label

    let _: <() as InternalFirst>::Output;
    //~^ ERROR internal message
    //~| NOTE stable label
}
