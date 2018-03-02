// rustfmt-reorder_imports: true
// rustfmt-normalize_comments: true

use d; use c; use b; use a; 
// The previous line has a space after the `use a;` 

mod a { use d; use c; use b; use a; }

use z;

use y;



use x;
use a;