//@ check-pass

pub struct VecNumber<'s> {
    pub vec_number: Vec<Number<'s>>,
    pub auxiliary_object: &'s Vec<usize>,
}

pub struct Number<'s> {
    pub number: &'s usize,
}

impl<'s> VecNumber<'s> {
    pub fn vec_number_iterable_per_item_in_auxiliary_object(
        &self,
    ) -> impl Iterator<Item = (&'s usize, impl Iterator<Item = &Number<'s>>)> {
        self.auxiliary_object.iter().map(move |n| {
            let iter_number = self.vec_number.iter();
            (n, iter_number)
        })
    }
}

fn main() {}
