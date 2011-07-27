// xfail-stage0
//error-pattern:no clauses match

fn main() {
    #macro([#trivial, 1 * 2 * 4 * 2 * 1]);

    assert (#trivial(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16) ==
                16);
}