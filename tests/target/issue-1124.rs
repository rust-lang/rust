// rustfmt-reorder_imports: true
// rustfmt-normalize_comments: true

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

use a;
use x;
use y;
use z;
