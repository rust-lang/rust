// rustfmt-indent_style: Block
// Function arguments layout

fn lorem() {}

fn lorem(ipsum: usize) {}

fn lorem(ipsum: usize, dolor: usize, sit: usize, amet: usize, consectetur: usize, adipiscing: usize, elit: usize) {
    // body
}

// #1441
extern "system" {
    pub fn GetConsoleHistoryInfo(console_history_info: *mut ConsoleHistoryInfo) -> Boooooooooooooool;
}

// rustfmt should not add trailing comma for variadic function. See #1623.
extern "C" {
    pub fn variadic_fn(first_parameter: FirstParameterType,
                       second_parameter: SecondParameterType,
                       ...);
}

// #1652
fn deconstruct(foo: Bar) -> (SocketAddr, Header, Method, RequestUri, HttpVersion, AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA) {
}
