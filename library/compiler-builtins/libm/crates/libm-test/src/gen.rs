//! Different generators that can create random or systematic bit patterns.

use crate::GenerateInput;
pub mod domain_logspace;
pub mod edge_cases;
pub mod random;

/// A wrapper to turn any iterator into an `ExactSizeIterator`. Asserts the final result to ensure
/// the provided size was correct.
#[derive(Debug)]
pub struct KnownSize<I> {
    total: u64,
    current: u64,
    iter: I,
}

impl<I> KnownSize<I> {
    pub fn new(iter: I, total: u64) -> Self {
        Self { total, current: 0, iter }
    }
}

impl<I: Iterator> Iterator for KnownSize<I> {
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        let next = self.iter.next();
        if next.is_some() {
            self.current += 1;
            return next;
        }

        assert_eq!(self.current, self.total, "total items did not match expected");
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = usize::try_from(self.total - self.current).unwrap();
        (remaining, Some(remaining))
    }
}

impl<I: Iterator> ExactSizeIterator for KnownSize<I> {}

/// Helper type to turn any reusable input into a generator.
#[derive(Clone, Debug, Default)]
pub struct CachedInput {
    pub inputs_f32: Vec<(f32, f32, f32)>,
    pub inputs_f64: Vec<(f64, f64, f64)>,
    pub inputs_i32: Vec<(i32, i32, i32)>,
}

impl GenerateInput<(f32,)> for CachedInput {
    fn get_cases(&self) -> impl Iterator<Item = (f32,)> {
        self.inputs_f32.iter().map(|f| (f.0,))
    }
}

impl GenerateInput<(f32, f32)> for CachedInput {
    fn get_cases(&self) -> impl Iterator<Item = (f32, f32)> {
        self.inputs_f32.iter().map(|f| (f.0, f.1))
    }
}

impl GenerateInput<(i32, f32)> for CachedInput {
    fn get_cases(&self) -> impl Iterator<Item = (i32, f32)> {
        self.inputs_i32.iter().zip(self.inputs_f32.iter()).map(|(i, f)| (i.0, f.0))
    }
}

impl GenerateInput<(f32, i32)> for CachedInput {
    fn get_cases(&self) -> impl Iterator<Item = (f32, i32)> {
        GenerateInput::<(i32, f32)>::get_cases(self).map(|(i, f)| (f, i))
    }
}

impl GenerateInput<(f32, f32, f32)> for CachedInput {
    fn get_cases(&self) -> impl Iterator<Item = (f32, f32, f32)> {
        self.inputs_f32.iter().copied()
    }
}

impl GenerateInput<(f64,)> for CachedInput {
    fn get_cases(&self) -> impl Iterator<Item = (f64,)> {
        self.inputs_f64.iter().map(|f| (f.0,))
    }
}

impl GenerateInput<(f64, f64)> for CachedInput {
    fn get_cases(&self) -> impl Iterator<Item = (f64, f64)> {
        self.inputs_f64.iter().map(|f| (f.0, f.1))
    }
}

impl GenerateInput<(i32, f64)> for CachedInput {
    fn get_cases(&self) -> impl Iterator<Item = (i32, f64)> {
        self.inputs_i32.iter().zip(self.inputs_f64.iter()).map(|(i, f)| (i.0, f.0))
    }
}

impl GenerateInput<(f64, i32)> for CachedInput {
    fn get_cases(&self) -> impl Iterator<Item = (f64, i32)> {
        GenerateInput::<(i32, f64)>::get_cases(self).map(|(i, f)| (f, i))
    }
}

impl GenerateInput<(f64, f64, f64)> for CachedInput {
    fn get_cases(&self) -> impl Iterator<Item = (f64, f64, f64)> {
        self.inputs_f64.iter().copied()
    }
}
