// rustfmt-normalize_comments: true
// rustfmt-wrap_comments: true
// rustfmt-hard_tabs: true

fn main() {
	let x = Bar;

	let y = Foo { a: x };

	Foo {
		a: foo(), // comment
		// comment
		b: bar(),
		..something
	};

	fn foo(a: i32, a: i32, a: i32, a: i32, a: i32, a: i32, a: i32, a: i32, a: i32, a: i32, a: i32) {
	}

	let str = "AAAAAAAAAAAAAAaAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAaAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAaAa";

	if let (
		some_very_large,
		tuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuple,
	) = 1 + 2 + 3
	{}

	if cond() {
		something();
	} else if different_cond() {
		something_else();
	} else {
		aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
			+ aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
	}

	unsafe /* very looooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooong
	        * comment */ {
	}

	unsafe /* So this is a very long comment.
	        * Multi-line, too.
	        * Will it still format correctly? */ {
	}

	let chain = funktion_kall()
		.go_to_next_line_with_tab()
		.go_to_next_line_with_tab()
		.go_to_next_line_with_tab();

	let z = [
		xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx,
		yyyyyyyyyyyyyyyyyyyyyyyyyyy,
		zzzzzzzzzzzzzzzzzz,
		q,
	];

	fn generic<T>(arg: T) -> &SomeType
	where
		T: Fn(
			// First arg
			A,
			// Second argument
			B,
			C,
			D,
			// pre comment
			E, // last comment
		) -> &SomeType,
	{
		arg(a, b, c, d, e)
	}

	loong_func().quux(move || if true { 1 } else { 2 });

	fffffffffffffffffffffffffffffffffff(a, {
		SCRIPT_TASK_ROOT.with(|root| {
			*root.borrow_mut() = Some(&script_task);
		});
	});
	a.b.c.d();

	x().y(|| match cond() {
		true => (),
		false => (),
	});
}

// #2296
impl Foo {
	// a comment
	// on multiple lines
	fn foo() {
		// another comment
		// on multiple lines
		let x = true;
	}
}
