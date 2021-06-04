// rustfmt-reorder_imports: true
// Reorder imports

use lorem;
use ipsum;
use dolor;
use sit;

fn foo() {
    use C;
    use B;
    use A;

    bar();

    use F;
    use E;
    use D;
}
