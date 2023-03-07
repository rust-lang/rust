// Regresison test for issue #75361
// Tests that we don't ICE on mismatched types with inference variables


trait MyTrait {
    type Item;
}

pub trait Graph {
  type EdgeType;

  fn adjacent_edges(&self) -> Box<dyn MyTrait<Item = &Self::EdgeType>>;
}

impl<T> Graph for T {
  type EdgeType = T;

  fn adjacent_edges(&self) -> Box<dyn MyTrait<Item = &Self::EdgeType> + '_> { //~ ERROR `impl`
      panic!()
  }

}

fn main() {}
