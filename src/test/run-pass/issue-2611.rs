use iter::BaseIter;

trait FlatMapToVec<A> {
  fn flat_map_to_vec<B:copy, IB:BaseIter<B>>(op: fn(A) -> IB) -> ~[B];
}

impl<A:copy> BaseIter<A>: FlatMapToVec<A> {
   fn flat_map_to_vec<B:copy, IB:BaseIter<B>>(op: fn(A) -> IB) -> ~[B] {
     iter::flat_map_to_vec(self, op)
   }
}

fn main() {}