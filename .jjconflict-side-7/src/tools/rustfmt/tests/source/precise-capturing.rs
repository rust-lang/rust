fn hello() -> impl
use<'a> + Sized {}

fn all_three() -> impl Sized + use<'a> + 'a;

fn pathological() -> impl use<'a, 'a, 'a, 'a, 'a, 'a, 'a, 'a, 'a, 'a, 'a, 'a, 'a, 'a, 'a,
'a, 'a, 'a, 'a, 'a, 'a, 'a, 'a, 'a,
'a, 'a, 'a, 'a, 'a, 'a, 'a, 'a, 'a, 'a, 'a, 'a, 'a, 'a, 
'a, 'a, 'a, 'a, 'a, 'a, 'a, 'a, 'a, 'a> + Sized {}
