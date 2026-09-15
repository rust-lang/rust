//! Regression test for https://github.com/rust-lang/rust/issues/86562.

//@ check-pass

fn row_groups() -> Vec<Vec<usize>> {
    let mut groups = Vec::new();

    for row in 0..5 {
        for group in &mut groups {
            if group[0] == row {
                group.push(row);
            }
        }
        groups.push(vec![row]);
    }

    groups
}

fn main() {
    drop(row_groups());
}
