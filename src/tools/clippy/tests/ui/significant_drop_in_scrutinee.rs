// FIXME: Ideally these suggestions would be fixed via rustfix. Blocked by rust-lang/rust#53934
//@no-rustfix
#![warn(clippy::significant_drop_in_scrutinee)]
#![allow(dead_code, unused_assignments)]
#![allow(
    clippy::match_single_binding,
    clippy::single_match,
    clippy::uninlined_format_args,
    clippy::needless_lifetimes
)]

use std::num::ParseIntError;
use std::ops::Deref;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, RwLock};

struct State {}

impl State {
    fn foo(&self) -> bool {
        true
    }

    fn bar(&self) {}
}

fn should_not_trigger_lint_with_mutex_guard_outside_match() {
    let mutex = Mutex::new(State {});

    // Should not trigger lint because the temporary should drop at the `;` on line before the match
    let is_foo = mutex.lock().unwrap().foo();
    match is_foo {
        true => {
            mutex.lock().unwrap().bar();
        },
        false => {},
    };
}

fn should_not_trigger_lint_with_mutex_guard_when_taking_ownership_in_match() {
    let mutex = Mutex::new(State {});

    // Should not trigger lint because the scrutinee is explicitly returning the MutexGuard,
    // so its lifetime should not be surprising.
    match mutex.lock() {
        Ok(guard) => {
            guard.foo();
            mutex.lock().unwrap().bar();
        },
        _ => {},
    };
}

fn should_trigger_lint_with_mutex_guard_in_match_scrutinee() {
    let mutex = Mutex::new(State {});

    // Should trigger lint because the lifetime of the temporary MutexGuard is surprising because it
    // is preserved until the end of the match, but there is no clear indication that this is the
    // case.
    match mutex.lock().unwrap().foo() {
        //~^ significant_drop_in_scrutinee
        true => {
            mutex.lock().unwrap().bar();
        },
        false => {},
    };
}

fn should_not_trigger_lint_with_mutex_guard_in_match_scrutinee_when_lint_allowed() {
    let mutex = Mutex::new(State {});

    // Lint should not be triggered because it is "allowed" below.
    #[allow(clippy::significant_drop_in_scrutinee)]
    match mutex.lock().unwrap().foo() {
        true => {
            mutex.lock().unwrap().bar();
        },
        false => {},
    };
}

fn should_not_trigger_lint_for_insignificant_drop() {
    // Should not trigger lint because there are no temporaries whose drops have a significant
    // side effect.
    match 1u64.to_string().is_empty() {
        true => {
            println!("It was empty")
        },
        false => {
            println!("It was not empty")
        },
    }
}

struct StateWithMutex {
    m: Mutex<u64>,
}

struct MutexGuardWrapper<'a> {
    mg: MutexGuard<'a, u64>,
}

impl<'a> MutexGuardWrapper<'a> {
    fn get_the_value(&self) -> u64 {
        *self.mg.deref()
    }
}

struct MutexGuardWrapperWrapper<'a> {
    mg: MutexGuardWrapper<'a>,
}

impl<'a> MutexGuardWrapperWrapper<'a> {
    fn get_the_value(&self) -> u64 {
        *self.mg.mg.deref()
    }
}

impl StateWithMutex {
    fn lock_m(&self) -> MutexGuardWrapper<'_> {
        MutexGuardWrapper {
            mg: self.m.lock().unwrap(),
        }
    }

    fn lock_m_m(&self) -> MutexGuardWrapperWrapper<'_> {
        MutexGuardWrapperWrapper {
            mg: MutexGuardWrapper {
                mg: self.m.lock().unwrap(),
            },
        }
    }

    fn foo(&self) -> bool {
        true
    }

    fn bar(&self) {}
}

fn should_trigger_lint_with_wrapped_mutex() {
    let s = StateWithMutex { m: Mutex::new(1) };

    // Should trigger lint because a temporary contains a type with a significant drop and its
    // lifetime is not obvious. Additionally, it is not obvious from looking at the scrutinee that
    // the temporary contains such a type, making it potentially even more surprising.
    match s.lock_m().get_the_value() {
        //~^ significant_drop_in_scrutinee
        1 => {
            println!("Got 1. Is it still 1?");
            println!("{}", s.lock_m().get_the_value());
        },
        2 => {
            println!("Got 2. Is it still 2?");
            println!("{}", s.lock_m().get_the_value());
        },
        _ => {},
    }
    println!("All done!");
}

fn should_trigger_lint_with_double_wrapped_mutex() {
    let s = StateWithMutex { m: Mutex::new(1) };

    // Should trigger lint because a temporary contains a type which further contains a type with a
    // significant drop and its lifetime is not obvious. Additionally, it is not obvious from
    // looking at the scrutinee that the temporary contains such a type, making it potentially even
    // more surprising.
    match s.lock_m_m().get_the_value() {
        //~^ significant_drop_in_scrutinee
        1 => {
            println!("Got 1. Is it still 1?");
            println!("{}", s.lock_m().get_the_value());
        },
        2 => {
            println!("Got 2. Is it still 2?");
            println!("{}", s.lock_m().get_the_value());
        },
        _ => {},
    }
    println!("All done!");
}

struct Counter {
    i: AtomicU64,
}

#[clippy::has_significant_drop]
struct CounterWrapper<'a> {
    counter: &'a Counter,
}

impl<'a> CounterWrapper<'a> {
    fn new(counter: &Counter) -> CounterWrapper<'_> {
        counter.i.fetch_add(1, Ordering::Relaxed);
        CounterWrapper { counter }
    }
}

impl<'a> Drop for CounterWrapper<'a> {
    fn drop(&mut self) {
        self.counter.i.fetch_sub(1, Ordering::Relaxed);
    }
}

impl Counter {
    fn temp_increment(&self) -> Vec<CounterWrapper<'_>> {
        vec![CounterWrapper::new(self), CounterWrapper::new(self)]
    }
}

fn should_trigger_lint_for_vec() {
    let counter = Counter { i: AtomicU64::new(0) };

    // Should trigger lint because the temporary in the scrutinee returns a collection of types
    // which have significant drops. The types with significant drops are also non-obvious when
    // reading the expression in the scrutinee.
    match counter.temp_increment().len() {
        //~^ significant_drop_in_scrutinee
        2 => {
            let current_count = counter.i.load(Ordering::Relaxed);
            println!("Current count {}", current_count);
            assert_eq!(current_count, 0);
        },
        1 => {},
        3 => {},
        _ => {},
    };
}

struct StateWithField {
    s: String,
}

// Should trigger lint only on the type in the tuple which is created using a temporary
// with a significant drop. Additionally, this test ensures that the format of the tuple
// is preserved correctly in the suggestion.
fn should_trigger_lint_for_tuple_in_scrutinee() {
    let mutex1 = Mutex::new(StateWithField { s: "one".to_owned() });

    {
        match (mutex1.lock().unwrap().s.len(), true) {
            //~^ significant_drop_in_scrutinee
            (3, _) => {
                println!("started");
                mutex1.lock().unwrap().s.len();
                println!("done");
            },
            (_, _) => {},
        };

        match (true, mutex1.lock().unwrap().s.len(), true) {
            //~^ significant_drop_in_scrutinee
            (_, 3, _) => {
                println!("started");
                mutex1.lock().unwrap().s.len();
                println!("done");
            },
            (_, _, _) => {},
        };

        let mutex2 = Mutex::new(StateWithField { s: "two".to_owned() });
        match (mutex1.lock().unwrap().s.len(), true, mutex2.lock().unwrap().s.len()) {
            //~^ significant_drop_in_scrutinee
            //~| significant_drop_in_scrutinee
            (3, _, 3) => {
                println!("started");
                mutex1.lock().unwrap().s.len();
                mutex2.lock().unwrap().s.len();
                println!("done");
            },
            (_, _, _) => {},
        };
    }
}

// Should not trigger lint since `String::as_str` returns a reference (i.e., `&str`)
// to the locked data (i.e., the `String`) and it is not surprising that matching such
// a reference needs to keep the data locked until the end of the match block.
fn should_not_trigger_lint_for_string_as_str() {
    let mutex1 = Mutex::new(StateWithField { s: "one".to_owned() });

    {
        let mutex2 = Mutex::new(StateWithField { s: "two".to_owned() });
        let mutex3 = Mutex::new(StateWithField { s: "three".to_owned() });

        match mutex3.lock().unwrap().s.as_str() {
            "three" => {
                println!("started");
                mutex1.lock().unwrap().s.len();
                mutex2.lock().unwrap().s.len();
                println!("done");
            },
            _ => {},
        };

        match (true, mutex3.lock().unwrap().s.as_str()) {
            (_, "three") => {
                println!("started");
                mutex1.lock().unwrap().s.len();
                mutex2.lock().unwrap().s.len();
                println!("done");
            },
            (_, _) => {},
        };
    }
}

// Should trigger lint when either side of a binary operation creates a temporary with a
// significant drop.
// To avoid potential unnecessary copies or creating references that would trigger the significant
// drop problem, the lint recommends moving the entire binary operation.
fn should_trigger_lint_for_accessing_field_in_mutex_in_one_side_of_binary_op() {
    let mutex = Mutex::new(StateWithField { s: "state".to_owned() });

    match mutex.lock().unwrap().s.len() > 1 {
        //~^ significant_drop_in_scrutinee
        true => {
            mutex.lock().unwrap().s.len();
        },
        false => {},
    };

    match 1 < mutex.lock().unwrap().s.len() {
        //~^ significant_drop_in_scrutinee
        true => {
            mutex.lock().unwrap().s.len();
        },
        false => {},
    };
}

// Should trigger lint when both sides of a binary operation creates a temporary with a
// significant drop.
// To avoid potential unnecessary copies or creating references that would trigger the significant
// drop problem, the lint recommends moving the entire binary operation.
fn should_trigger_lint_for_accessing_fields_in_mutex_in_both_sides_of_binary_op() {
    let mutex1 = Mutex::new(StateWithField { s: "state".to_owned() });
    let mutex2 = Mutex::new(StateWithField {
        s: "statewithfield".to_owned(),
    });

    match mutex1.lock().unwrap().s.len() < mutex2.lock().unwrap().s.len() {
        //~^ significant_drop_in_scrutinee
        //~| significant_drop_in_scrutinee
        true => {
            println!(
                "{} < {}",
                mutex1.lock().unwrap().s.len(),
                mutex2.lock().unwrap().s.len()
            );
        },
        false => {},
    };

    match mutex1.lock().unwrap().s.len() >= mutex2.lock().unwrap().s.len() {
        //~^ significant_drop_in_scrutinee
        //~| significant_drop_in_scrutinee
        true => {
            println!(
                "{} >= {}",
                mutex1.lock().unwrap().s.len(),
                mutex2.lock().unwrap().s.len()
            );
        },
        false => {},
    };
}

fn should_not_trigger_lint_for_closure_in_scrutinee() {
    let mutex1 = Mutex::new(StateWithField { s: "one".to_owned() });

    let get_mutex_guard = || mutex1.lock().unwrap().s.len();

    // Should not trigger lint because the temporary with a significant drop will be dropped
    // at the end of the closure, so the MutexGuard will be unlocked and not have a potentially
    // surprising lifetime.
    match get_mutex_guard() > 1 {
        true => {
            mutex1.lock().unwrap().s.len();
        },
        false => {},
    };
}

fn should_trigger_lint_for_return_from_closure_in_scrutinee() {
    let mutex1 = Mutex::new(StateWithField { s: "one".to_owned() });

    let get_mutex_guard = || mutex1.lock().unwrap();

    // Should trigger lint because the temporary with a significant drop is returned from the
    // closure but not used directly in any match arms, so it has a potentially surprising lifetime.
    match get_mutex_guard().s.len() > 1 {
        //~^ significant_drop_in_scrutinee
        true => {
            mutex1.lock().unwrap().s.len();
        },
        false => {},
    };
}

fn should_trigger_lint_for_return_from_match_in_scrutinee() {
    let mutex1 = Mutex::new(StateWithField { s: "one".to_owned() });
    let mutex2 = Mutex::new(StateWithField { s: "two".to_owned() });

    let i = 100;

    // Should trigger lint because the nested match within the scrutinee returns a temporary with a
    // significant drop is but not used directly in any match arms, so it has a potentially
    // surprising lifetime.
    match match i {
        //~^ significant_drop_in_scrutinee
        100 => mutex1.lock().unwrap(),
        _ => mutex2.lock().unwrap(),
    }
    .s
    .len()
        > 1
    {
        true => {
            mutex1.lock().unwrap().s.len();
        },
        false => {
            println!("nothing to do here");
        },
    };
}

fn should_trigger_lint_for_return_from_if_in_scrutinee() {
    let mutex1 = Mutex::new(StateWithField { s: "one".to_owned() });
    let mutex2 = Mutex::new(StateWithField { s: "two".to_owned() });

    let i = 100;

    // Should trigger lint because the nested if-expression within the scrutinee returns a temporary
    // with a significant drop is but not used directly in any match arms, so it has a potentially
    // surprising lifetime.
    match if i > 1 {
        //~^ significant_drop_in_scrutinee
        mutex1.lock().unwrap()
    } else {
        mutex2.lock().unwrap()
    }
    .s
    .len()
        > 1
    {
        true => {
            mutex1.lock().unwrap().s.len();
        },
        false => {},
    };
}

fn should_not_trigger_lint_for_if_in_scrutinee() {
    let mutex = Mutex::new(StateWithField { s: "state".to_owned() });

    let i = 100;

    // Should not trigger the lint because the temporary with a significant drop *is* dropped within
    // the body of the if-expression nested within the match scrutinee, and therefore does not have
    // a potentially surprising lifetime.
    match if i > 1 {
        mutex.lock().unwrap().s.len() > 1
    } else {
        false
    } {
        true => {
            mutex.lock().unwrap().s.len();
        },
        false => {},
    };
}

struct StateWithBoxedMutexGuard {
    u: Mutex<u64>,
}

impl StateWithBoxedMutexGuard {
    fn new() -> StateWithBoxedMutexGuard {
        StateWithBoxedMutexGuard { u: Mutex::new(42) }
    }
    fn lock(&self) -> Box<MutexGuard<'_, u64>> {
        Box::new(self.u.lock().unwrap())
    }
}

fn should_trigger_lint_for_boxed_mutex_guard() {
    let s = StateWithBoxedMutexGuard::new();

    // Should trigger lint because a temporary Box holding a type with a significant drop in a match
    // scrutinee may have a potentially surprising lifetime.
    match s.lock().deref().deref() {
        //~^ significant_drop_in_scrutinee
        0 | 1 => println!("Value was less than 2"),
        _ => println!("Value is {}", s.lock().deref()),
    };
}

struct StateStringWithBoxedMutexGuard {
    s: Mutex<String>,
}

impl StateStringWithBoxedMutexGuard {
    fn new() -> StateStringWithBoxedMutexGuard {
        StateStringWithBoxedMutexGuard {
            s: Mutex::new("A String".to_owned()),
        }
    }
    fn lock(&self) -> Box<MutexGuard<'_, String>> {
        Box::new(self.s.lock().unwrap())
    }
}

fn should_not_trigger_lint_for_string_ref() {
    let s = StateStringWithBoxedMutexGuard::new();

    let matcher = String::from("A String");

    // Should not trigger lint because the second `deref` returns a string reference (`&String`).
    // So it is not surprising that matching the reference implies that the lock needs to be held
    // until the end of the block.
    match s.lock().deref().deref() {
        matcher => println!("Value is {}", s.lock().deref()),
        _ => println!("Value was not a match"),
    };
}

struct StateWithIntField {
    i: u64,
}

// Should trigger lint when either side of an assign expression contains a temporary with a
// significant drop, because the temporary's lifetime will be extended to the end of the match.
// To avoid potential unnecessary copies or creating references that would trigger the significant
// drop problem, the lint recommends moving the entire binary operation.
fn should_trigger_lint_in_assign_expr() {
    let mutex = Mutex::new(StateWithIntField { i: 10 });

    let mut i = 100;

    match mutex.lock().unwrap().i = i {
        //~^ significant_drop_in_scrutinee
        _ => {
            println!("{}", mutex.lock().unwrap().i);
        },
    };

    match i = mutex.lock().unwrap().i {
        //~^ significant_drop_in_scrutinee
        _ => {
            println!("{}", mutex.lock().unwrap().i);
        },
    };

    match mutex.lock().unwrap().i += 1 {
        //~^ significant_drop_in_scrutinee
        _ => {
            println!("{}", mutex.lock().unwrap().i);
        },
    };

    match i += mutex.lock().unwrap().i {
        //~^ significant_drop_in_scrutinee
        _ => {
            println!("{}", mutex.lock().unwrap().i);
        },
    };
}

#[derive(Debug)]
enum RecursiveEnum {
    Foo(Option<Box<RecursiveEnum>>),
}

#[derive(Debug)]
enum GenericRecursiveEnum<T> {
    Foo(T, Option<Box<GenericRecursiveEnum<T>>>),
}

fn should_not_cause_stack_overflow() {
    // Test that when a type recursively contains itself, a stack overflow does not occur when
    // checking sub-types for significant drops.
    let f = RecursiveEnum::Foo(Some(Box::new(RecursiveEnum::Foo(None))));
    match f {
        RecursiveEnum::Foo(Some(f)) => {
            println!("{:?}", f)
        },
        RecursiveEnum::Foo(f) => {
            println!("{:?}", f)
        },
    }

    let f = GenericRecursiveEnum::Foo(1u64, Some(Box::new(GenericRecursiveEnum::Foo(2u64, None))));
    match f {
        GenericRecursiveEnum::Foo(i, Some(f)) => {
            println!("{} {:?}", i, f)
        },
        GenericRecursiveEnum::Foo(i, f) => {
            println!("{} {:?}", i, f)
        },
    }
}

fn should_not_produce_lint_for_try_desugar() -> Result<u64, ParseIntError> {
    // TryDesugar (i.e. using `?` for a Result type) will turn into a match but is out of scope
    // for this lint
    let rwlock = RwLock::new("1".to_string());
    let result = rwlock.read().unwrap().parse::<u64>()?;
    println!("{}", result);
    rwlock.write().unwrap().push('2');
    Ok(result)
}

struct ResultReturner {
    s: String,
}

impl ResultReturner {
    fn to_number(&self) -> Result<i64, ParseIntError> {
        self.s.parse::<i64>()
    }
}

fn should_trigger_lint_for_non_ref_move_and_clone_suggestion() {
    let rwlock = RwLock::<ResultReturner>::new(ResultReturner { s: "1".to_string() });
    match rwlock.read().unwrap().to_number() {
        //~^ significant_drop_in_scrutinee
        Ok(n) => println!("Converted to number: {}", n),
        Err(e) => println!("Could not convert {} to number", e),
    };
}

fn should_not_trigger_lint_for_read_write_lock_for_loop() {
    let rwlock = RwLock::<Vec<String>>::new(vec!["1".to_string()]);
    // Should not trigger lint. Since we're iterating over the data, it's obvious that the lock
    // has to be held until the iteration finishes.
    // https://github.com/rust-lang/rust-clippy/issues/8987
    for s in rwlock.read().unwrap().iter() {
        println!("{}", s);
    }
}

fn do_bar(mutex: &Mutex<State>) {
    mutex.lock().unwrap().bar();
}

fn should_trigger_lint_without_significant_drop_in_arm() {
    let mutex = Mutex::new(State {});

    // Should trigger lint because the lifetime of the temporary MutexGuard is surprising because it
    // is preserved until the end of the match, but there is no clear indication that this is the
    // case.
    match mutex.lock().unwrap().foo() {
        //~^ significant_drop_in_scrutinee
        true => do_bar(&mutex),
        false => {},
    };
}

fn should_not_trigger_on_significant_iterator_drop() {
    let lines = std::io::stdin().lines();
    for line in lines {
        println!("foo: {}", line.unwrap());
    }
}

// https://github.com/rust-lang/rust-clippy/issues/9072
fn should_not_trigger_lint_if_place_expr_has_significant_drop() {
    let x = Mutex::new(vec![1, 2, 3]);
    let x_guard = x.lock().unwrap();

    match x_guard[0] {
        1 => println!("1!"),
        x => println!("{x}"),
    }

    match x_guard.len() {
        1 => println!("1!"),
        x => println!("{x}"),
    }
}

struct Guard<'a, T>(MutexGuard<'a, T>);

struct Ref<'a, T>(&'a T);

impl<'a, T> Guard<'a, T> {
    fn guard(&self) -> &MutexGuard<'_, T> {
        &self.0
    }

    fn guard_ref(&self) -> Ref<'_, MutexGuard<'_, T>> {
        Ref(&self.0)
    }

    fn take(self) -> Box<MutexGuard<'a, T>> {
        Box::new(self.0)
    }
}

fn should_not_trigger_for_significant_drop_ref() {
    let mutex = Mutex::new(vec![1, 2]);
    let guard = Guard(mutex.lock().unwrap());

    match guard.guard().len() {
        0 => println!("empty"),
        _ => println!("not empty"),
    }

    match guard.guard_ref().0.len() {
        0 => println!("empty"),
        _ => println!("not empty"),
    }

    match guard.take().len() {
        //~^ significant_drop_in_scrutinee
        0 => println!("empty"),
        _ => println!("not empty"),
    };
}

struct Foo<'a>(&'a Vec<u32>);

impl<'a> Foo<'a> {
    fn copy_old_lifetime(&self) -> &'a Vec<u32> {
        self.0
    }

    fn reborrow_new_lifetime(&self) -> &Vec<u32> {
        self.0
    }
}

fn should_trigger_lint_if_and_only_if_lifetime_is_irrelevant() {
    let vec = Vec::new();
    let mutex = Mutex::new(Foo(&vec));

    // Should trigger lint even if `copy_old_lifetime()` has a lifetime, as the lifetime of
    // `&vec` is unrelated to the temporary with significant drop (i.e., the `MutexGuard`).
    for val in mutex.lock().unwrap().copy_old_lifetime() {
        //~^ significant_drop_in_scrutinee
        println!("{}", val);
    }

    // Should not trigger lint because `reborrow_new_lifetime()` has a lifetime and the
    // lifetime is related to the temporary with significant drop (i.e., the `MutexGuard`).
    for val in mutex.lock().unwrap().reborrow_new_lifetime() {
        println!("{}", val);
    }
}

fn should_not_trigger_lint_for_complex_lifetime() {
    let mutex = Mutex::new(vec!["hello".to_owned()]);
    let string = "world".to_owned();

    // Should not trigger lint due to the relevant lifetime.
    for c in mutex.lock().unwrap().first().unwrap().chars() {
        println!("{}", c);
    }

    // Should trigger lint due to the irrelevant lifetime.
    //
    // FIXME: The lifetime is too complex to analyze. In order to avoid false positives, we do not
    // trigger lint.
    for c in mutex.lock().unwrap().first().map(|_| &string).unwrap().chars() {
        println!("{}", c);
    }
}

fn should_not_trigger_lint_with_explicit_drop() {
    let mutex = Mutex::new(vec![1]);

    // Should not trigger lint since the drop explicitly happens.
    for val in [drop(mutex.lock().unwrap()), ()] {
        println!("{:?}", val);
    }

    // Should trigger lint if there is no explicit drop.
    for val in [mutex.lock().unwrap()[0], 2] {
        //~^ significant_drop_in_scrutinee
        println!("{:?}", val);
    }
}

fn should_trigger_lint_in_if_let() {
    let mutex = Mutex::new(vec![1]);

    if let Some(val) = mutex.lock().unwrap().first().copied() {
        //~^ significant_drop_in_scrutinee
        println!("{}", val);
    }

    // Should not trigger lint without the final `copied()`, because we actually hold a reference
    // (i.e., the `val`) to the locked data.
    if let Some(val) = mutex.lock().unwrap().first() {
        println!("{}", val);
    };
}

fn should_trigger_lint_in_while_let() {
    let mutex = Mutex::new(vec![1]);

    while let Some(val) = mutex.lock().unwrap().pop() {
        //~^ significant_drop_in_scrutinee
        println!("{}", val);
    }
}

async fn foo_async(mutex: &Mutex<i32>) -> Option<MutexGuard<'_, i32>> {
    Some(mutex.lock().unwrap())
}

async fn should_trigger_lint_for_async(mutex: Mutex<i32>) -> i32 {
    match *foo_async(&mutex).await.unwrap() {
        //~^ significant_drop_in_scrutinee
        n if n < 10 => n,
        _ => 10,
    }
}

async fn should_not_trigger_lint_in_async_expansion(mutex: Mutex<i32>) -> i32 {
    match foo_async(&mutex).await {
        Some(guard) => *guard,
        _ => 0,
    }
}

fn should_trigger_lint_in_match_expr() {
    let mutex = Mutex::new(State {});

    // Should trigger lint because the lifetime of the temporary MutexGuard is surprising because it
    // is preserved until the end of the match, but there is no clear indication that this is the
    // case.
    let _ = match mutex.lock().unwrap().foo() {
        //~^ significant_drop_in_scrutinee
        true => 0,
        false => 1,
    };
}

fn main() {}
