fn print() {
    println!(cfg_select! {
        unix => { "unix" }
        _ => { "not " +           "unix" }
    });

    println!(cfg_select! {
        unix => { "unix" }
        _ => { "not unix" }
    });
}

std::cfg_select! {
    target_arch = "aarch64" => {
        use std::sync::OnceCell;

        fn foo() {
                return 3;
        }

    }
    _ =>                     {
        compile_error!("mal",   "formed")
    }
    false => {
        compile_error!("also",        "mal",   "formed")
    }
}

core::cfg_select! {
    windows => {}
    unix => {                 }
    _ => {}
}

fn nested_blocks() {
    println!(cfg_select! {
        unix => {{ "unix"} }
        _ => {
            {  { "not " +           "unix"
            } } }
    });
}

cfg_select! {}

cfg_select! {
    any(true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true) => {}
    all(target_arch = "x86_64", true, target_endian = "little", debug_assertions, panic = "unwind", target_env = "gnu") => {}
    all(any(target_arch = "x86_64", true, target_endian = "little"), debug_assertions, panic = "unwind", all(target_env = "gnu", true)) => {}

    any(true, true, true, true, true, true, true, true, true, true,
        true, true, true, true, true, true, true, true, true, true) => {}
    all(
        any(target_arch = "x86_64", true, target_endian = "little"), debug_assertions,
        panic = "unwind", all(target_env = "gnu", true)
    ) => {}

    all(
        any(target_arch = "x86_64", true, target_endian = "little"),
        debug_assertions,
        panic = "unwind",
        all(target_env = "gnu", true)
    ) => {}

    // This line is under 80 characters, no reason to break.
    any(feature = "acdefg", true, true, true, true, true, true, true, true) => {
        compile_error!("foo")
    }
    // The cfg is under 80 characters, but the line as a whole is over 80 characters.
    any(feature = "acdefgh123", true, true, true, true, true, true, true, true) => {
        compile_error!("foo")
    }
    // The cfg is over 80 characters.
    any(feature = "acdefgh1234", true, true, true, true, true, true, true, true) => {
        compile_error!("foo")
    }

    _ => {}
}

// When there is no way to make the line fit, formatting bails.
cfg_select! {
    feature = "debug-with-rustfmt-long-long-long-long-loooooooonnnnnnnnnnnnnnnggggggffffffffffffffff"
    => {
        // abc
        println!();
    }
    feature = "debug-with-rustfmt-long-long-long-long-loooooooonnnnnnnnnnnnnnnggggggffffffffffffffff"
        => {
        // abc
    }
    anything("some other long long long long long thing long long long long long long long long long long long") => {
    }
}

// Unfortunately comments are dropped.
cfg_select! {
    _ => { /* a comment */ }
    false => {
        /* a comment */
        { 1 }
    }
}
