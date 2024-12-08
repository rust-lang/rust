fn a() {
    let mut v = vec![1, 2, 3];
    let vb: &mut [isize] = &mut v;
    match vb {
        &mut [_a, ref tail @ ..] => {
            v.push(tail[0] + tail[1]); //~ ERROR cannot borrow
        }
        _ => {}
    };
}

fn main() {}
