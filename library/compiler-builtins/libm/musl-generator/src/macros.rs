macro_rules! f32 {
    ($($fun:ident,)+) => {{
        $(
            // check type signature
            let _: fn(f32) -> f32 = libm::$fun;
            let mut $fun = File::create(concat!("bin/output/musl.", stringify!($fun)))?;
        )+

        for x in shared::F32.iter() {
            $(
                let y = unsafe {
                    extern "C" {
                        fn $fun(_: f32) -> f32;
                    }

                    $fun(*x)
                };

                $fun.write_all(&y.to_bits().to_bytes())?;
            )+
        }
    }};
}

macro_rules! f32f32 {
    ($($fun:ident,)+) => {{
        $(
            // check type signature
            let _: fn(f32, f32) -> f32 = libm::$fun;
            let mut $fun = File::create(concat!("bin/output/musl.", stringify!($fun)))?;
        )+

        for (x0, x1) in shared::F32F32.iter() {
            $(
                let y = unsafe {
                    extern "C" {
                        fn $fun(_: f32, _: f32) -> f32;
                    }

                    $fun(*x0, *x1)
                };

                $fun.write_all(&y.to_bits().to_bytes())?;
            )+
        }
    }};
}

macro_rules! f32f32f32 {
    ($($fun:ident,)+) => {{
        $(
            // check type signature
            let _: fn(f32, f32, f32) -> f32 = libm::$fun;
            let mut $fun = File::create(concat!("bin/output/musl.", stringify!($fun)))?;
        )+

            for (x0, x1, x2) in shared::F32F32F32.iter() {
                $(
                    let y = unsafe {
                        extern "C" {
                            fn $fun(_: f32, _: f32, _: f32) -> f32;
                        }

                        $fun(*x0, *x1, *x2)
                    };

                    $fun.write_all(&y.to_bits().to_bytes())?;
                )+
            }
    }};
}

macro_rules! f32i32 {
    ($($fun:ident,)+) => {{
        $(
            // check type signature
            let _: fn(f32, i32) -> f32 = libm::$fun;
            let mut $fun = File::create(concat!("bin/output/musl.", stringify!($fun)))?;
        )+

            for (x0, x1) in shared::F32I32.iter() {
                $(
                    let y = unsafe {
                        extern "C" {
                            fn $fun(_: f32, _: i32) -> f32;
                        }

                        $fun(*x0, *x1 as i32)
                    };

                    $fun.write_all(&y.to_bits().to_bytes())?;
                )+
            }
    }};
}

macro_rules! f64 {
    ($($fun:ident,)+) => {{
        $(
            // check type signature
            let _: fn(f64) -> f64 = libm::$fun;
            let mut $fun = File::create(concat!("bin/output/musl.", stringify!($fun)))?;
        )+

        for x in shared::F64.iter() {
            $(
                let y = unsafe {
                    extern "C" {
                        fn $fun(_: f64) -> f64;
                    }

                    $fun(*x)
                };

                $fun.write_all(&y.to_bits().to_bytes())?;
            )+
        }
    }};
}

macro_rules! f64f64 {
    ($($fun:ident,)+) => {{
        $(
            // check type signature
            let _: fn(f64, f64) -> f64 = libm::$fun;
            let mut $fun = File::create(concat!("bin/output/musl.", stringify!($fun)))?;
        )+

        for (x0, x1) in shared::F64F64.iter() {
            $(
                let y = unsafe {
                    extern "C" {
                        fn $fun(_: f64, _: f64) -> f64;
                    }

                    $fun(*x0, *x1)
                };

                $fun.write_all(&y.to_bits().to_bytes())?;
            )+
        }
    }};
}

macro_rules! f64f64f64 {
    ($($fun:ident,)+) => {{
        $(
            // check type signature
            let _: fn(f64, f64, f64) -> f64 = libm::$fun;
            let mut $fun = File::create(concat!("bin/output/musl.", stringify!($fun)))?;
        )+

            for (x0, x1, x2) in shared::F64F64F64.iter() {
                $(
                    let y = unsafe {
                        extern "C" {
                            fn $fun(_: f64, _: f64, _: f64) -> f64;
                        }

                        $fun(*x0, *x1, *x2)
                    };

                    $fun.write_all(&y.to_bits().to_bytes())?;
                )+
            }
    }};
}

macro_rules! f64i32 {
    ($($fun:ident,)+) => {{
        $(
            // check type signature
            let _: fn(f64, i32) -> f64 = libm::$fun;
            let mut $fun = File::create(concat!("bin/output/musl.", stringify!($fun)))?;
        )+

            for (x0, x1) in shared::F64I32.iter() {
                $(
                    let y = unsafe {
                        extern "C" {
                            fn $fun(_: f64, _: i32) -> f64;
                        }

                        $fun(*x0, *x1 as i32)
                    };

                    $fun.write_all(&y.to_bits().to_bytes())?;
                )+
            }
    }};
}
