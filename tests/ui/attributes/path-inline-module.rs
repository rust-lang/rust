// ERROR: #[path] on inline module with NO external submodules
#[path = "foo.rs"]
mod inline_module {} //~ ERROR attribute `#[path]` is useless on inline modules

// ERROR: #[path] on inline module with only inline submodules
#[path = "foo.rs"]
mod inline_with_inline_sub { //~ ERROR attribute `#[path]` is useless on inline modules
    mod inner {} // inline, not external
}

// ERROR: #![path] inside module with no external submodules
mod useless_inner { //~ ERROR attribute `#[path]` is useless here as there are no nested external modules
    #![path = "some_dir"]
    // no external submodules
}

// ERROR: #![path] inside module where submodule is inline (has body)
mod thread_inline_sub { //~ ERROR attribute `#[path]` is useless here as there are no nested external modules
    #![path = "thread_files"]
    #[path = "tls.rs"]
    mod local_data {} //~ ERROR attribute `#[path]` is useless on inline modules
}

fn main() {}
