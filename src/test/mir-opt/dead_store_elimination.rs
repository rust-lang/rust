#[inline(never)]
pub fn read<T>(_: T) {}

// Check that we make the optimization in the cases we're interested in
// EMIT_MIR dead_store_elimination.coverage.SimpleLocalDse.diff
pub fn coverage() {
    let mut x = 0;
    x = 1;
    x = 2;
    read(x);

    let mut x = 10;
    let mut r = &mut 11;
    x = *r;
    x = 12;
    read(x);

    struct S(u32);

    let mut s = S(20);
    s.0 = 21;
    s.0 = 22;
    read(s);

    let mut s = S(30);
    s.0 = 31;
    s = S(32);
    read(s);

    let mut x = 40;
    x = 41;
    // StorageDead here
}

// EMIT_MIR dead_store_elimination.indexing.SimpleLocalDse.diff
pub fn indexing() {
    let mut i = 0;
    let a = [1; 2];
    let x = a[i];
    read(x);

    let mut i = 0;
    let a = [11; 2];
    let r = &a;
    let r = &r[i];
    read(r);

    let mut i = 0;
    let mut a = [13; 2];
    a[i] = 14;
    read(a);
}

// Don't optimize out things when memory accesses may observe them
// EMIT_MIR dead_store_elimination.memory.SimpleLocalDse.diff
pub fn memory() {
    let mut x = 0;
    let mut y = 1; // don't optimize out
    let r = &mut y;
    x = *r;
    y = 2;
    read(x);
}

// EMIT_MIR dead_store_elimination.find_reads.SimpleLocalDse.diff
pub fn find_reads() {
    let mut a = 0;
    let mut b = 1;
    let c = a + b;
    read(c);

    let mut a = 10;
    let r = &a;
    read(r);

    let mut a = 20;
    let e = Some(a);
    read(e);

    let mut a = Some(30);
    if let Some(x) = a {
    } else {
    }
}

fn main() {
    coverage();
    indexing();
    memory();
    find_reads();
}
