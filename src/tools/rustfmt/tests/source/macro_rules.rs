// rustfmt-format_macro_matchers: true

macro_rules! m {
    () => ();
    ( $ x : ident ) => ();
    ( $ m1 : ident , $ m2 : ident , $ x : ident ) => ();
    ( $($beginning:ident),*;$middle:ident;$($end:ident),* ) => ();
    ( $($beginning: ident),*; $middle: ident; $($end: ident),*; $($beginning: ident),*; $middle: ident; $($end: ident),* ) => {};
    ( $ name : ident ( $ ( $ dol : tt $ var : ident ) * ) $ ( $ body : tt ) * ) => ();
    ( $( $ i : ident : $ ty : ty , $def : expr , $stb : expr , $ ( $ dstring : tt ) , + ) ; + $ ( ; ) *
      $( $ i : ident : $ ty : ty , $def : expr , $stb : expr , $ ( $ dstring : tt ) , + ) ; + $ ( ; ) *
    ) => {};
    ( $foo: tt foo [$ attr : meta] $name: ident ) => {};
    ( $foo: tt [$ attr: meta] $name: ident ) => {};
    ( $foo: tt &'a [$attr : meta] $name: ident ) => {};
    ( $foo: tt foo # [ $attr : meta] $name: ident ) => {};
    ( $foo: tt # [ $attr : meta] $name: ident) => {};
    ( $foo: tt &'a # [ $attr : meta] $name: ident ) => {};
    ( $ x : tt foo bar foo bar foo bar $ y : tt => x*y*z $ z : tt , $ ( $a: tt ) , * ) => {};
}


macro_rules! impl_a_method {
    ($n:ident ( $a:ident : $ta:ty ) -> $ret:ty { $body:expr }) => {
        fn $n($a:$ta) -> $ret { $body }
        macro_rules! $n { ($va:expr) => { $n($va) } }
    };
    ($n:ident ( $a:ident : $ta:ty, $b:ident : $tb:ty ) -> $ret:ty { $body:expr }) => {
        fn $n($a:$ta, $b:$tb) -> $ret { $body }
        macro_rules! $n { ($va:expr, $vb:expr) => { $n($va, $vb) } }
    };
    ($n:ident ( $a:ident : $ta:ty, $b:ident : $tb:ty, $c:ident : $tc:ty ) -> $ret:ty { $body:expr }) => {
        fn $n($a:$ta, $b:$tb, $c:$tc) -> $ret { $body }
        macro_rules! $n { ($va:expr, $vb:expr, $vc:expr) => { $n($va, $vb, $vc) } }
    };
    ($n:ident ( $a:ident : $ta:ty, $b:ident : $tb:ty, $c:ident : $tc:ty, $d:ident : $td:ty ) -> $ret:ty { $body:expr }) => {
        fn $n($a:$ta, $b:$tb, $c:$tc, $d:$td) -> $ret { $body }
        macro_rules! $n { ($va:expr, $vb:expr, $vc:expr, $vd:expr) => { $n($va, $vb, $vc, $vd) } }
    };
}

macro_rules! m {
	// a
	($expr :expr,  $( $func : ident    ) *   ) => {
		{
		let    x =    $expr;
									                $func (
														x
											)
	}
	};

				/* b */

   	()           => {/* c */};

		 				(@tag)   =>
						 {

						 };

// d
( $item:ident  ) =>      {
	mod macro_item    {  struct $item ; }
};
}

macro m2 {
	// a
	($expr :expr,  $( $func : ident    ) *   ) => {
		{
		let    x =    $expr;
									                $func (
														x
											)
	}
	}

				/* b */

   	()           => {/* c */}

		 				(@tag)   =>
						 {

						 }

// d
( $item:ident  ) =>      {
	mod macro_item    {  struct $item ; }
}
}

// #2438, #2476
macro_rules! m {
    () => {
        fn foo() {
            this_line_is_98_characters_long_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx(
            );
        }
    }
}
macro_rules! m {
    () => {
        fn foo() {
            this_line_is_99_characters_long_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx(
);
        }
    };
}
macro_rules! m {
    () => {
        fn foo() {
            this_line_is_100_characters_long_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx(
);
        }
    };
}
macro_rules! m {
    () => {
        fn foo() {
            this_line_is_101_characters_long_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx(
            );
        }
    };
}

// #2439
macro_rules! m {
    ($line0_xxxxxxxxxxxxxxxxx: expr, $line1_xxxxxxxxxxxxxxxxx: expr, $line2_xxxxxxxxxxxxxxxxx: expr, $line3_xxxxxxxxxxxxxxxxx: expr,) => {};
}

// #2466
// Skip formatting `macro_rules!` that are not using `{}`.
macro_rules! m (
    () => ()
);
macro_rules! m [
    () => ()
];

// #2470
macro foo($type_name: ident, $docs: expr) {
    #[allow(non_camel_case_types)]
    #[doc=$docs]
    #[derive(Debug, Clone, Copy)]
    pub struct $type_name;
}

// #2534
macro_rules! foo {
    ($a:ident : $b:ty) => {};
    ($a:ident $b:ident $c:ident) => {};
}

// #2538
macro_rules! add_message_to_notes {
    ($msg:expr) => {{
        let mut lines = message.lines();
        notes.push_str(&format!("\n{}: {}", level, lines.next().unwrap()));
        for line in lines {
            notes.push_str(&format!(
                "\n{:indent$}{line}",
                "",
                indent = level.len() + 2,
                line = line,
            ));
        }
    }}
}

// #2560
macro_rules! binary {
    ($_self:ident,$expr:expr, $lhs:expr,$func:ident) => {
        while $_self.matched($expr) {
            let op = $_self.get_binary_op()?;

            let rhs = Box::new($_self.$func()?);

           $lhs = Spanned {
                span: $lhs.get_span().to(rhs.get_span()),
                value: Expression::Binary {
                    lhs: Box::new($lhs),
                    op,
                    rhs,
                },
            }
        }
    };
}

// #2558
macro_rules! m {
    ($x:) => {};
    ($($foo:expr)()?) => {};
}

// #2749
macro_rules! foo {
    ($(x)* {}) => {};
    ($(x)* ()) => {};
    ($(x)* []) => {};
}
macro_rules! __wundergraph_expand_sqlite_mutation {
    ( $mutation_name:ident $((context = $($context:tt)*))*{ $( $entity_name:ident( $(insert = $insert:ident,)* $(update = $update:ident,)* $(delete = $($delete:tt)+)* ), )* } ) => {};
}

// #2607
macro_rules! bench {
    ($ty:ident) => {
        criterion_group!(
            name = benches;
            config = ::common_bench::reduced_samples();
            targets = call, map;
        );
    };
}

// #2770
macro_rules! save_regs {
    () => {
        asm!("push rax
              push rcx
              push rdx
              push rsi
              push rdi
              push r8
              push r9
              push r10
              push r11"
             :::: "intel", "volatile");
    };
}

// #2721
macro_rules! impl_as_byte_slice_arrays {
    ($n:expr,) => {};
    ($n:expr, $N:ident, $($NN:ident,)*) => {
        impl_as_byte_slice_arrays!($n - 1, $($NN,)*);
        
        impl<T> AsByteSliceMut for [T; $n] where [T]: AsByteSliceMut {
            fn as_byte_slice_mut(&mut self) -> &mut [u8] {
                self[..].as_byte_slice_mut()
            }

            fn to_le(&mut self) {
                self[..].to_le()
            }
        }
    };
    (!div $n:expr,) => {};
    (!div $n:expr, $N:ident, $($NN:ident,)*) => {
        impl_as_byte_slice_arrays!(!div $n / 2, $($NN,)*);

        impl<T> AsByteSliceMut for [T; $n] where [T]: AsByteSliceMut {
            fn as_byte_slice_mut(&mut self) -> &mut [u8] {
                self[..].as_byte_slice_mut()
            }
            
            fn to_le(&mut self) {
                self[..].to_le()
            }
        }
    };
}

// #2919
fn foo() {
    {
        macro_rules! touch_value {
            ($func:ident, $value:expr) => {{
                let result = API::get_cached().$func(self, key.as_ptr(), $value, ffi::VSPropAppendMode::paTouch);
                let result = API::get_cached().$func(self, key.as_ptr(), $value, ffi::VSPropAppend);
                let result = API::get_cached().$func(self, key.as_ptr(), $value, ffi::VSPropAppendM);
                let result = APIIIIIIIII::get_cached().$func(self, key.as_ptr(), $value, ffi::VSPropAppendM);
                let result = API::get_cached().$func(self, key.as_ptr(), $value, ffi::VSPropAppendMMMMMMMMMM);
                debug_assert!(result == 0);
            }};
        }
    }
}

// #2642
macro_rules! template {
    ($name: expr) => {
        format_args!(r##"
"http://example.com"

# test
"##, $name)
    }
}

macro_rules! template {
    () => {
        format_args!(r"
//

")
    }
}
