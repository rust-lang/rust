// rustfmt-use_try_shorthand: true

fn main() {
    let x = try!(some_expr());

    let y = try!(a.very.loooooooooooooooooooooooooooooooooooooong().chain().inside().weeeeeeeeeeeeeee()).test().0.x;
}

fn test() {
    a?
}

fn issue1291() {
    try!(fs::create_dir_all(&gitfiledir).chain_err(|| {
        format!("failed to create the {} submodule directory for the workarea",
                name)
    }));
}
