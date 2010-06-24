// -*- rust -*-

type colour = tag(red(), green(), blue());
type tree = tag(children(@list), leaf(colour));
type list = tag(cons(@tree, @list), nil());

type small_list = tag(kons(int,@small_list), neel());

fn main() {
}

