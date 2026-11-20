// rustfmt-reorder_imports: true

use a;
use b;
use c;
use d;
// The previous line has a space after the `use a;`

mod a {
    use a;
    use b;
    use c;
    use d;
}

use z;

use y;

use a;
use x;
