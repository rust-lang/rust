use std::cell::Cell;
use std::marker::PhantomData;
use std::pin::Pin;
use std::ops::{Generator, GeneratorState};

#[derive(Copy, Clone)]
pub struct AccessAction(*mut dyn FnMut());

impl AccessAction {
    pub fn get(self) -> *mut dyn FnMut() {
        self.0
    }
}

#[derive(Copy, Clone)]
pub enum Action {
    Access(AccessAction),
    Complete,
}

thread_local!(pub static BOX_REGION_ARG: Cell<Action> = Cell::new(Action::Complete));

pub struct PinnedGenerator<I, A, R> {
    generator: Pin<Box<dyn Generator<Yield = YieldType<I, A>, Return = R>>>
}

impl<I, A, R> PinnedGenerator<I, A, R> {
    pub fn new<
        T: Generator<Yield = YieldType<I, A>, Return = R> + 'static
    >(generator: T) -> (I, Self) {
        let mut result = PinnedGenerator {
            generator: Box::pin(generator)
        };

        // Run it to the first yield to set it up
        let init = match Pin::new(&mut result.generator).resume() {
            GeneratorState::Yielded(
                YieldType::Initial(y)
            ) => y,
            _ => panic!()
        };

        (init, result)
    }

    pub unsafe fn access(&mut self, closure: *mut dyn FnMut()) {
        BOX_REGION_ARG.with(|i| {
            i.set(Action::Access(AccessAction(closure)));
        });

        // Call the generator, which in turn will call the closure in BOX_REGION_ARG
        if let GeneratorState::Complete(_) = Pin::new(&mut self.generator).resume() {
            panic!()
        }
    }

    pub fn complete(&mut self) -> R {
        // Tell the generator we want it to complete, consuming it and yielding a result
        BOX_REGION_ARG.with(|i| {
            i.set(Action::Complete)
        });

        let result = Pin::new(&mut self.generator).resume();
        if let GeneratorState::Complete(r) = result {
            r
        } else {
            panic!()
        }
    }
}

#[derive(PartialEq)]
pub struct Marker<T>(PhantomData<T>);

impl<T> Marker<T> {
    pub unsafe fn new() -> Self {
        Marker(PhantomData)
    }
}

pub enum YieldType<I, A> {
    Initial(I),
    Accessor(Marker<A>),
}

#[macro_export]
#[allow_internal_unstable(fn_traits)]
macro_rules! declare_box_region_type {
    (impl $v:vis
     $name: ident,
     $yield_type:ty,
     for($($lifetimes:tt)*),
     ($($args:ty),*) -> ($reti:ty, $retc:ty)
    ) => {
        $v struct $name($crate::box_region::PinnedGenerator<
            $reti,
            for<$($lifetimes)*> fn(($($args,)*)),
            $retc
        >);

        impl $name {
            fn new<T: ::std::ops::Generator<Yield = $yield_type, Return = $retc> + 'static>(
                generator: T
            ) -> ($reti, Self) {
                let (initial, pinned) = $crate::box_region::PinnedGenerator::new(generator);
                (initial, $name(pinned))
            }

            $v fn access<F: for<$($lifetimes)*> FnOnce($($args,)*) -> R, R>(&mut self, f: F) -> R {
                // Turn the FnOnce closure into *mut dyn FnMut()
                // so we can pass it in to the generator using the BOX_REGION_ARG thread local
                let mut r = None;
                let mut f = Some(f);
                let mut_f: &mut dyn for<$($lifetimes)*> FnMut(($($args,)*)) =
                    &mut |args| {
                        let f = f.take().unwrap();
                        r = Some(FnOnce::call_once(f, args));
                };
                let mut_f = mut_f as *mut dyn for<$($lifetimes)*> FnMut(($($args,)*));

                // Get the generator to call our closure
                unsafe {
                    self.0.access(::std::mem::transmute(mut_f));
                }

                // Unwrap the result
                r.unwrap()
            }

            $v fn complete(mut self) -> $retc {
                self.0.complete()
            }

            fn initial_yield(value: $reti) -> $yield_type {
                $crate::box_region::YieldType::Initial(value)
            }
        }
    };

    ($v:vis $name: ident, for($($lifetimes:tt)*), ($($args:ty),*) -> ($reti:ty, $retc:ty)) => {
        declare_box_region_type!(
            impl $v $name,
            $crate::box_region::YieldType<$reti, for<$($lifetimes)*> fn(($($args,)*))>,
            for($($lifetimes)*),
            ($($args),*) -> ($reti, $retc)
        );
    };
}

#[macro_export]
#[allow_internal_unstable(fn_traits)]
macro_rules! box_region_allow_access {
    (for($($lifetimes:tt)*), ($($args:ty),*), ($($exprs:expr),*) ) => {
        loop {
            match $crate::box_region::BOX_REGION_ARG.with(|i| i.get()) {
                $crate::box_region::Action::Access(accessor) => {
                    let accessor: &mut dyn for<$($lifetimes)*> FnMut($($args),*) = unsafe {
                        ::std::mem::transmute(accessor.get())
                    };
                    (*accessor)(($($exprs),*));
                    unsafe {
                        let marker = $crate::box_region::Marker::<
                            for<$($lifetimes)*> fn(($($args,)*))
                        >::new();
                        yield $crate::box_region::YieldType::Accessor(marker)
                    };
                }
                $crate::box_region::Action::Complete => break,
            }
        }
    }
}
