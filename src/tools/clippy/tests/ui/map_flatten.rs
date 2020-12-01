// run-rustfix

#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::let_underscore_drop)]
#![allow(clippy::missing_docs_in_private_items)]
#![allow(clippy::map_identity)]
#![allow(clippy::unnecessary_wraps)]

fn main() {
    // mapping to Option on Iterator
    fn option_id(x: i8) -> Option<i8> {
        Some(x)
    }
    let option_id_ref: fn(i8) -> Option<i8> = option_id;
    let option_id_closure = |x| Some(x);
    let _: Vec<_> = vec![5_i8; 6].into_iter().map(option_id).flatten().collect();
    let _: Vec<_> = vec![5_i8; 6].into_iter().map(option_id_ref).flatten().collect();
    let _: Vec<_> = vec![5_i8; 6].into_iter().map(option_id_closure).flatten().collect();
    let _: Vec<_> = vec![5_i8; 6].into_iter().map(|x| x.checked_add(1)).flatten().collect();

    // mapping to Iterator on Iterator
    let _: Vec<_> = vec![5_i8; 6].into_iter().map(|x| 0..x).flatten().collect();

    // mapping to Option on Option
    let _: Option<_> = (Some(Some(1))).map(|x| x).flatten();
}
