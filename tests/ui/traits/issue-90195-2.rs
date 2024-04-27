//@ check-pass
pub trait Archive {
    type Archived;
}

impl<T> Archive for Option<T> {
    type Archived = ();
}
pub type Archived<T> = <T as Archive>::Archived;

pub trait Deserialize<D> {}

const ARRAY_SIZE: usize = 32;
impl<__D> Deserialize<__D> for ()
where
    Option<[u8; ARRAY_SIZE]>: Archive,
    Archived<Option<[u8; ARRAY_SIZE]>>: Deserialize<__D>,
{
}
fn main() {}
