//@ check-pass

fn take_edge_counters(
    x: &mut Option<Vec<i32>>,
) -> Option<impl Iterator<Item = i32>> {
    x.take().map_or(None, |m| Some(m.into_iter()))
}

fn main() {}
