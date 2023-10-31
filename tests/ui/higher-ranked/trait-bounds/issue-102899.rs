// check-pass

pub trait BufferTrait<'buffer> {
    type Subset<'channel>
    where
        'buffer: 'channel;

    fn for_each_subset<F>(&self, f: F)
    where
        F: for<'channel> Fn(Self::Subset<'channel>);
}

pub struct SomeBuffer<'buffer> {
    samples: &'buffer [()],
}

impl<'buffer> BufferTrait<'buffer> for SomeBuffer<'buffer> {
    type Subset<'subset> = Subset<'subset> where 'buffer: 'subset;

    fn for_each_subset<F>(&self, _f: F)
    where
        F: for<'subset> Fn(Subset<'subset>),
    {
        todo!()
    }
}

pub struct Subset<'subset> {
    buffer: &'subset [()],
}

fn main() {}
