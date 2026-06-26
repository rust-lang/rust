// Test the errors resulting from meta-variables hitting EOF.

#![crate_type = "lib"]
#![feature(macro_guard_matcher)]

macro_rules! unambig {
    (item      $x:item      ) => {};
    (block     $x:block     ) => {};
    (stmt      $x:stmt      ) => {};
    (pat       $x:pat       ) => {};
    (pat_param $x:pat_param ) => {};
    (expr      $x:expr      ) => {};
    (expr_2021 $x:expr_2021 ) => {};
    (ty        $x:ty        ) => {};
    (ident     $x:ident     ) => {};
    (lifetime  $x:lifetime  ) => {};
    (literal   $x:literal   ) => {};
    (meta      $x:meta      ) => {};
    (path      $x:path      ) => {};
    (vis       $x:vis       ) => {};
    (guard     $x:guard     ) => {};
    (tt        $x:tt        ) => {};
}

unambig!(item /* eof */); //~ ERROR unexpected end of macro invocation
unambig!(block /* eof */); //~ ERROR unexpected end of macro invocation
unambig!(stmt /* eof */); //~ ERROR unexpected end of macro invocation
unambig!(pat /* eof */); //~ ERROR unexpected end of macro invocation
unambig!(pat_param /* eof */); //~ ERROR unexpected end of macro invocation
unambig!(expr /* eof */); //~ ERROR unexpected end of macro invocation
unambig!(expr_2021 /* eof */); //~ ERROR unexpected end of macro invocation
unambig!(ty /* eof */); //~ ERROR unexpected end of macro invocation
unambig!(ident /* eof */); //~ ERROR unexpected end of macro invocation
unambig!(lifetime /* eof */); //~ ERROR unexpected end of macro invocation
unambig!(literal /* eof */); //~ ERROR unexpected end of macro invocation
unambig!(meta /* eof */); //~ ERROR unexpected end of macro invocation
unambig!(path /* eof */); //~ ERROR unexpected end of macro invocation
unambig!(vis /* eof */); //~ ERROR unexpected end of macro invocation
unambig!(guard /* eof */); //~ ERROR unexpected end of macro invocation
unambig!(tt /* eof */); //~ ERROR unexpected end of macro invocation
