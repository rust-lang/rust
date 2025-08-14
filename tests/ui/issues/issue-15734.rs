//@ run-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

use std::ops::Index;

struct Mat<T> { data: Vec<T>, cols: usize, }

impl<T> Mat<T> {
    fn new(data: Vec<T>, cols: usize) -> Mat<T> {
        Mat { data: data, cols: cols }
    }
    fn row<'a>(&'a self, row: usize) -> Row<&'a Mat<T>> {
        Row { mat: self, row: row, }
    }
}

impl<T> Index<(usize, usize)> for Mat<T> {
    type Output = T;

    fn index<'a>(&'a self, (row, col): (usize, usize)) -> &'a T {
        &self.data[row * self.cols + col]
    }
}

impl<'a, T> Index<(usize, usize)> for &'a Mat<T> {
    type Output = T;

    fn index<'b>(&'b self, index: (usize, usize)) -> &'b T {
        (*self).index(index)
    }
}

struct Row<M> { mat: M, row: usize, }

impl<T, M: Index<(usize, usize), Output=T>> Index<usize> for Row<M> {
    type Output = T;

    fn index<'a>(&'a self, col: usize) -> &'a T {
        &self.mat[(self.row, col)]
    }
}

fn main() {
    let m = Mat::new(vec![1, 2, 3, 4, 5, 6], 3);
    let r = m.row(1);

    assert_eq!(r.index(2), &6);
    assert_eq!(r[2], 6);
    assert_eq!(r[2], 6);
    assert_eq!(6, r[2]);

    let e = r[2];
    assert_eq!(e, 6);

    let e: usize = r[2];
    assert_eq!(e, 6);
}
