pub struct Heap;

pub struct FakeHeap;

pub struct FakeVec<T, A = FakeHeap> { pub f: Option<(T,A)> }
