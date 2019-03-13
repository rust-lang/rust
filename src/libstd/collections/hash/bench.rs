#![cfg(test)]

use test::Bencher;

#[bench]
fn new_drop(b: &mut Bencher) {
    use super::map::HashMap;

    b.iter(|| {
        let m: HashMap<i32, i32> = HashMap::new();
        assert_eq!(m.len(), 0);
    })
}

#[bench]
fn new_insert_drop(b: &mut Bencher) {
    use super::map::HashMap;

    b.iter(|| {
        let mut m = HashMap::new();
        m.insert(0, 0);
        assert_eq!(m.len(), 1);
    })
}

#[bench]
fn grow_by_insertion(b: &mut Bencher) {
    use super::map::HashMap;

    let mut m = HashMap::new();

    for i in 1..1001 {
        m.insert(i, i);
    }

    let mut k = 1001;

    b.iter(|| {
        m.insert(k, k);
        k += 1;
    });
}

#[bench]
fn find_existing(b: &mut Bencher) {
    use super::map::HashMap;

    let mut m = HashMap::new();

    for i in 1..1001 {
        m.insert(i, i);
    }

    b.iter(|| {
        for i in 1..1001 {
            m.contains_key(&i);
        }
    });
}

#[bench]
fn find_nonexisting(b: &mut Bencher) {
    use super::map::HashMap;

    let mut m = HashMap::new();

    for i in 1..1001 {
        m.insert(i, i);
    }

    b.iter(|| {
        for i in 1001..2001 {
            m.contains_key(&i);
        }
    });
}

#[bench]
fn hashmap_as_queue(b: &mut Bencher) {
    use super::map::HashMap;

    let mut m = HashMap::new();

    for i in 1..1001 {
        m.insert(i, i);
    }

    let mut k = 1;

    b.iter(|| {
        m.remove(&k);
        m.insert(k + 1000, k + 1000);
        k += 1;
    });
}

#[bench]
fn get_remove_insert(b: &mut Bencher) {
    use super::map::HashMap;

    let mut m = HashMap::new();

    for i in 1..1001 {
        m.insert(i, i);
    }

    let mut k = 1;

    b.iter(|| {
        m.get(&(k + 400));
        m.get(&(k + 2000));
        m.remove(&k);
        m.insert(k + 1000, k + 1000);
        k += 1;
    })
}
