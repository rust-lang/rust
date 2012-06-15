// xfail-test
import iter;
import iter::base_iter;

impl Q<A> for base_iter<A> {
   fn flat_map_to_vec<B:copy, IB:base_iter<B>>(op: fn(B) -> IB) -> [B] {
      iter::flat_map_to_vec(self, op)
   }
}

fn main() {}