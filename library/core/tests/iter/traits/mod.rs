mod accum;
mod collect;
mod double_ended;
mod iterator;
mod marker;

#[test]
fn test_rposition() {
    fn f(xy: &(isize, char)) -> bool {
        let (_x, y) = *xy;
        y == 'b'
    }
    fn g(xy: &(isize, char)) -> bool {
        let (_x, y) = *xy;
        y == 'd'
    }
    let v = [(0, 'a'), (1, 'b'), (2, 'c'), (3, 'b')];

    assert_eq!(v.iter().rposition(f), Some(3));
    assert!(v.iter().rposition(g).is_none());
}

#[test]
fn test_rev_rposition() {
    let v = [0, 0, 1, 1];
    assert_eq!(v.iter().rev().rposition(|&x| x == 1), Some(1));
}

#[test]
#[should_panic]
fn test_rposition_panic() {
    let v: [(Box<_>, Box<_>); 4] = [(box 0, box 0), (box 0, box 0), (box 0, box 0), (box 0, box 0)];
    let mut i = 0;
    v.iter().rposition(|_elt| {
        if i == 2 {
            panic!()
        }
        i += 1;
        false
    });
}
