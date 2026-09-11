// rustfmt-indent_style: Visual

fn reflow_list_node_with_rule(node: &CompoundNode, rule: &Rule, args: &[Arg], shape: &Shape) where T: FOo, U: Bar {
    let mut effects = HashMap::new();
}

fn reflow_list_node_with_rule(node: &CompoundNode, rule: &Rule, args: &[Arg], shape: &Shape) where T: FOo {
    let mut effects = HashMap::new();
}

fn reflow_list_node_with_rule(node: &CompoundNode, rule: &Rule, args: &[Arg], shape: &Shape, shape: &Shape) where T: FOo, U: Bar {
    let mut effects = HashMap::new();
}

fn reflow_list_node_with_rule(node: &CompoundNode, rule: &Rule, args: &[Arg], shape: &Shape, shape: &Shape) where T: FOo {
    let mut effects = HashMap::new();
}

fn reflow_list_node_with_rule(node: &CompoundNode, rule: &Rule, args: &[Arg], shape: &Shape) -> Option<String> where T: FOo, U: Bar {
    let mut effects = HashMap::new();
}

fn reflow_list_node_with_rule(node: &CompoundNode, rule: &Rule, args: &[Arg], shape: &Shape) -> Option<String> where T: FOo {
    let mut effects = HashMap::new();
}

pub trait Test {
    fn very_long_method_name<F>(self, f: F) -> MyVeryLongReturnType where F: FnMut(Self::Item) -> bool;

    fn exactly_100_chars1<F>(self, f: F) -> MyVeryLongReturnType where F: FnMut(Self::Item) -> bool;
}

fn very_long_function_name<F>(very_long_argument: F) -> MyVeryLongReturnType where F: FnMut(Self::Item) -> bool { }

struct VeryLongTupleStructName<A, B, C, D, E>(LongLongTypename, LongLongTypename, i32, i32) where A: LongTrait;

struct Exactly100CharsToSemicolon<A, B, C, D, E>
    (LongLongTypename, i32, i32)
    where A: LongTrait1234;

struct AlwaysOnNextLine<LongLongTypename, LongTypename, A, B, C, D, E, F> where A: LongTrait {
    x: i32
}

pub trait SomeTrait<T>
    where
    T: Something + Sync + Send + Display     + Debug     + Copy + Hash + Debug + Display + Write + Read + FromStr
{
}

// #2020
impl<'a, 'gcx, 'tcx> ProbeContext<'a, 'gcx, 'tcx> {
    fn elaborate_bounds<F>(&mut self, bounds: &[ty::PolyTraitRef<'tcx>], mut mk_cand: F)
    where F: for<'b> FnMut(&mut ProbeContext<'b, 'gcx, 'tcx>, ty::PolyTraitRef<'tcx>, ty::AssociatedItem),
    {
        // ...
    }
}
