// rustfmt-style_edition: 2024
// rustfmt-single_line_let_else_max_width: 100

fn issue5901() {
    #[cfg(target_os = "linux")]
    let Some(x) = foo else { todo!() };

    #[cfg(target_os = "linux")]
    // Some comments between attributes and let-else statement
    let Some(x) = foo else { todo!() };

    #[cfg(target_os = "linux")]
    #[cfg(target_arch = "x86_64")]
    let Some(x) = foo else { todo!() };

    // The else block is multi-lined
    #[cfg(target_os = "linux")]
    let Some(x) = foo else { return; };

    // The else block will be single-lined because attributes and comments before `let`
    // are no longer included when calculating max width
    #[cfg(target_os = "linux")]
    #[cfg(target_arch = "x86_64")]
    // Some comments between attributes and let-else statement
    let Some(x) = foo else { todo!() };

    // Some more test cases for v2 formatting with attributes

    #[cfg(target_os = "linux")]
    #[cfg(target_arch = "x86_64")]
    let Some(x) = opt
    // pre else keyword line-comment
    else { return; };

    #[cfg(target_os = "linux")]
    #[cfg(target_arch = "x86_64")]
    let Some(x) = opt else
    // post else keyword line-comment
    { return; };

    #[cfg(target_os = "linux")]
    #[cfg(target_arch = "x86_64")]
    let Foo {x: Bar(..), y: FooBar(..), z: Baz(..)} = opt else {
        return;
    };

    #[cfg(target_os = "linux")]
    #[cfg(target_arch = "x86_64")]
    let Some(Ok((Message::ChangeColor(super::color::Color::Rgb(r, g, b)), Point { x, y, z }))) = opt else {
        return;
    };

    #[cfg(target_os = "linux")]
    #[cfg(target_arch = "x86_64")]
    let Some(x) = very_very_very_very_very_very_very_very_very_very_very_very_long_expression_in_assign_rhs() else { return; };
}
