#![warn(clippy::self_assignment)]
#![allow(clippy::useless_vec, clippy::needless_pass_by_ref_mut)]

pub struct S<'a> {
    a: i32,
    b: [i32; 10],
    c: Vec<Vec<i32>>,
    e: &'a mut i32,
    f: &'a mut i32,
}

pub fn positives(mut a: usize, b: &mut u32, mut s: S) {
    a = a;
    *b = *b;
    s = s;
    s.a = s.a;
    s.b[9] = s.b[5 + 4];
    s.c[0][1] = s.c[0][1];
    s.b[a] = s.b[a];
    *s.e = *s.e;
    s.b[a + 10] = s.b[10 + a];

    let mut t = (0, 1);
    t.1 = t.1;
    t.0 = (t.0);
}

pub fn negatives_not_equal(mut a: usize, b: &mut usize, mut s: S) {
    dbg!(&a);
    a = *b;
    dbg!(&a);
    s.b[1] += s.b[1];
    s.b[1] = s.b[2];
    s.c[1][0] = s.c[0][1];
    s.b[a] = s.b[*b];
    s.b[a + 10] = s.b[a + 11];
    *s.e = *s.f;

    let mut t = (0, 1);
    t.0 = t.1;
}

#[allow(clippy::mixed_read_write_in_expression)]
pub fn negatives_side_effects() {
    let mut v = vec![1, 2, 3, 4, 5];
    let mut i = 0;
    v[{
        i += 1;
        i
    }] = v[{
        i += 1;
        i
    }];

    fn next(n: &mut usize) -> usize {
        let v = *n;
        *n += 1;
        v
    }

    let mut w = vec![1, 2, 3, 4, 5];
    let mut i = 0;
    let i = &mut i;
    w[next(i)] = w[next(i)];
    w[next(i)] = w[next(i)];
}

fn main() {}
