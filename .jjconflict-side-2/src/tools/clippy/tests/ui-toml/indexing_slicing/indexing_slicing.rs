//@compile-flags: --test
#![warn(clippy::indexing_slicing)]
#![allow(clippy::no_effect)]

fn main() {
    let x = [1, 2, 3, 4];
    let index: usize = 1;
    &x[index..];
    //~^ indexing_slicing
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_fn() {
        let x = [1, 2, 3, 4];
        let index: usize = 1;
        &x[index..];
    }
}
