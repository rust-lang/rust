// rustfmt-fn_args_layout: Block
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

