//@ check-pass

fn nop(_s: &[& i32]) {}
fn nop_subslice(_s: &[i32]) {}

fn const_index_ok(s: &mut [i32]) {
    if let [ref first, ref second, _, ref fourth, ..] = *s {
        if let [_, _, ref mut third, ..] = *s {
            nop(&[first, second, third, fourth]);
        }
    }
}

fn const_index_from_end_ok(s: &mut [i32]) {
    if let [.., ref fourth, ref third, _, ref first] = *s {
        if let [.., ref mut second, _] = *s {
            nop(&[first, second, third, fourth]);
        }
    }
}

fn const_index_mixed(s: &mut [i32]) {
    if let [.., _, ref from_end4, ref from_end3, _, ref from_end1] = *s {
        if let [ref mut from_begin0, ..] = *s {
            nop(&[from_begin0, from_end1, from_end3, from_end4]);
        }
    }
    if let [ref from_begin0, ref from_begin1, _, ref from_begin3, _, ..] = *s {
        if let [.., ref mut from_end1] = *s {
            nop(&[from_begin0, from_begin1, from_begin3, from_end1]);
        }
    }
}

fn const_index_and_subslice_ok(s: &mut [i32]) {
    if let [ref first, ref second, ..] = *s {
        if let [_, _, ref mut tail @ ..] = *s {
            nop(&[first, second]);
            nop_subslice(tail);
        }
    }
}

fn const_index_and_subslice_from_end_ok(s: &mut [i32]) {
    if let [.., ref second, ref first] = *s {
        if let [ref mut tail @ .., _, _] = *s {
            nop(&[first, second]);
            nop_subslice(tail);
        }
    }
}

fn main() {
    let mut v = [1,2,3,4];
    const_index_ok(&mut v);
    const_index_from_end_ok(&mut v);
    const_index_and_subslice_ok(&mut v);
    const_index_and_subslice_from_end_ok(&mut v);
}
