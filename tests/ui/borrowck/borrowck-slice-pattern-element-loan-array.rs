fn nop(_s: &[& i32]) {}
fn nop_subslice(_s: &[i32]) {}

fn const_index_err(s: &mut [i32; 4]) {
    let [ref first, ref second, ..] = *s;
    let [_, ref mut  second2, ref mut third, ..] = *s; //~ERROR
    nop(&[first, second, second2, third]);
}

fn const_index_from_end_err(s: &mut [i32; 4]) {
    let [.., ref fourth, ref third, _, ref first] = *s;
    let [.., ref mut third2, _, _] = *s; //~ERROR
    nop(&[first, third, third2, fourth]);
}

fn const_index_mixed(s: &mut [i32; 6]) {
    let [.., _, ref from_end4, ref from_end3, _, ref from_end1] = *s;

    let [_, _, ref mut from_begin2, ..] = *s; //~ERROR
    nop(&[from_begin2, from_end1, from_end3, from_end4]);
    let [_, _, _, ref mut from_begin3, ..] = *s; //~ERROR
    nop(&[from_begin3, from_end1, from_end3, from_end4]);

    let [ref from_begin0, ref from_begin1, _, ref from_begin3, _, ..] = *s;

    let [.., ref mut from_end3, _,  _] = *s; //~ERROR
    nop(&[from_begin0, from_begin1, from_begin3, from_end3]);
}

fn const_index_and_subslice_err(s: &mut [i32; 4]) {
    let [ref first, ref second, ..] = *s;
    let [_, ref mut tail @ ..] = *s; //~ERROR
    nop(&[first, second]);
    nop_subslice(tail);
}

fn const_index_and_subslice_from_end_err(s: &mut [i32; 4]) {
    let [.., ref second, ref first] = *s;
    let [ref mut tail @ .., _] = *s; //~ERROR
    nop(&[first, second]);
    nop_subslice(tail);
}

fn subslices_overlap(s: &mut [i32; 4]) {
    let [_,  ref s1 @ ..] = *s;
    let [ref mut s2 @ .., _, _] = *s; //~ERROR
    nop_subslice(s1);
    nop_subslice(s2);
}

fn main() {
    let mut v = [1,2,3,4];
    const_index_err(&mut v);
    const_index_from_end_err(&mut v);
    const_index_and_subslice_err(&mut v);
    const_index_and_subslice_from_end_err(&mut v);
    subslices_overlap(&mut v);
}
