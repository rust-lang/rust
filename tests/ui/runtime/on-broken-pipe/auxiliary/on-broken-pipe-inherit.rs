// -Cprefer-dynamic is not supported by eii yet
//@ no-prefer-dynamic
#![crate_type = "rlib"]

#![feature(extern_item_impls)]
#![feature(on_broken_pipe)]

#[std::io::on_broken_pipe]
fn on_broken_pipe() -> std::io::OnBrokenPipe {
    std::io::OnBrokenPipe::Inherit
}
