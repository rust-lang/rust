// Test that the macro backtrace facility works (supporting file)

// a non-local macro
#[macro_export]
macro_rules! ping {
    () => {
        pong!();
    }
}

#[macro_export]
macro_rules! deep {
    () => {
        foo!();
    }
}

#[macro_export]
macro_rules! foo {
    () => {
        bar!();
    }
}

#[macro_export]
macro_rules! bar {
    () => {
        ping!();
    }
}
