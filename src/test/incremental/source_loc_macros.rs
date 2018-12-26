// This test makes sure that different expansions of the file!(), line!(),
// column!() macros get picked up by the incr. comp. hash.

// revisions:rpass1 rpass2

// compile-flags: -Z query-dep-graph

#![feature(rustc_attrs)]

#[rustc_clean(label="Hir", cfg="rpass2")]
#[rustc_clean(label="HirBody", cfg="rpass2")]
fn line_same() {
    let _ = line!();
}

#[rustc_clean(label="Hir", cfg="rpass2")]
#[rustc_clean(label="HirBody", cfg="rpass2")]
fn col_same() {
    let _ = column!();
}

#[rustc_clean(label="Hir", cfg="rpass2")]
#[rustc_clean(label="HirBody", cfg="rpass2")]
fn file_same() {
    let _ = file!();
}

#[rustc_clean(label="Hir", cfg="rpass2")]
#[rustc_dirty(label="HirBody", cfg="rpass2")]
fn line_different() {
    #[cfg(rpass1)]
    {
        let _ = line!();
    }
    #[cfg(rpass2)]
    {
        let _ = line!();
    }
}

#[rustc_clean(label="Hir", cfg="rpass2")]
#[rustc_dirty(label="HirBody", cfg="rpass2")]
fn col_different() {
    #[cfg(rpass1)]
    {
        let _ = column!();
    }
    #[cfg(rpass2)]
    {
        let _ =        column!();
    }
}

fn main() {
    line_same();
    line_different();
    col_same();
    col_different();
    file_same();
}
