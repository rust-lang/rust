// run-rustfix

#[warn(clippy::needless_option_as_deref)]

fn main() {
    // should lint
    let _: Option<&usize> = Some(&1).as_deref();
    let _: Option<&mut usize> = Some(&mut 1).as_deref_mut();

    // should not lint
    let _ = Some(Box::new(1)).as_deref();
    let _ = Some(Box::new(1)).as_deref_mut();
}
