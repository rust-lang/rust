// compile-flags:-Zpolymorphize=on -Zprint-mono-items=lazy -Copt-level=1
// ignore-tidy-linelength

#![crate_type = "rlib"]

// Test that only one copy of `Iter::map` and `iter::repeat` are generated.

fn unused<T>() -> u64 {
    42
}

fn foo<T>() {
    let x = [1, 2, 3, std::mem::size_of::<T>()];
    x.iter().map(|_| ());
}

//~ MONO_ITEM fn core::iter[0]::adapters[0]::{{impl}}[29]::new[0]<core::slice[0]::Iter[0]<usize>, pr_75255::foo[0]::{{closure}}[0]<T>> @@ pr_75255-cgu.0[External]
//~ MONO_ITEM fn core::iter[0]::traits[0]::iterator[0]::Iterator[0]::map[0]<core::slice[0]::Iter[0]<usize>, (), pr_75255::foo[0]::{{closure}}[0]<T>> @@ pr_75255-cgu.1[Internal]

fn bar<T>() {
    std::iter::repeat(unused::<T>);
}

//~ MONO_ITEM fn core::iter[0]::sources[0]::repeat[0]<fn() -> u64> @@ pr_75255-cgu.1[Internal]

pub fn dispatch() {
    foo::<String>();
    foo::<Vec<String>>();

    bar::<String>();
    bar::<Vec<String>>();
}

// These are all the items that aren't relevant to the test.
//~ MONO_ITEM fn core::mem[0]::size_of[0]<alloc::string[0]::String[0]> @@ pr_75255-cgu.1[Internal]
//~ MONO_ITEM fn core::mem[0]::size_of[0]<alloc::vec[0]::Vec[0]<alloc::string[0]::String[0]>> @@ pr_75255-cgu.1[Internal]
//~ MONO_ITEM fn core::mem[0]::size_of[0]<usize> @@ pr_75255-cgu.1[Internal]
//~ MONO_ITEM fn core::ptr[0]::const_ptr[0]::{{impl}}[0]::add[0]<usize> @@ pr_75255-cgu.1[Internal]
//~ MONO_ITEM fn core::ptr[0]::const_ptr[0]::{{impl}}[0]::is_null[0]<usize> @@ pr_75255-cgu.1[Internal]
//~ MONO_ITEM fn core::ptr[0]::const_ptr[0]::{{impl}}[0]::offset[0]<usize> @@ pr_75255-cgu.1[Internal]
//~ MONO_ITEM fn core::ptr[0]::const_ptr[0]::{{impl}}[0]::wrapping_add[0]<u8> @@ pr_75255-cgu.1[Internal]
//~ MONO_ITEM fn core::ptr[0]::const_ptr[0]::{{impl}}[0]::wrapping_offset[0]<u8> @@ pr_75255-cgu.1[Internal]
//~ MONO_ITEM fn core::ptr[0]::non_null[0]::{{impl}}[3]::new_unchecked[0]<usize> @@ pr_75255-cgu.1[Internal]
//~ MONO_ITEM fn core::ptr[0]::null[0]<u8> @@ pr_75255-cgu.1[Internal]
//~ MONO_ITEM fn core::slice[0]::{{impl}}[0]::as_ptr[0]<usize> @@ pr_75255-cgu.1[Internal]
//~ MONO_ITEM fn core::slice[0]::{{impl}}[0]::iter[0]<usize> @@ pr_75255-cgu.1[Internal]
//~ MONO_ITEM fn core::slice[0]::{{impl}}[0]::len[0]<usize> @@ pr_75255-cgu.1[Internal]
//~ MONO_ITEM fn pr_75255::dispatch[0] @@ pr_75255-cgu.1[External]
//~ MONO_ITEM fn pr_75255::foo[0]<alloc::string[0]::String[0]> @@ pr_75255-cgu.1[Internal]
//~ MONO_ITEM fn pr_75255::foo[0]<alloc::vec[0]::Vec[0]<alloc::string[0]::String[0]>> @@ pr_75255-cgu.1[Internal]
//~ MONO_ITEM fn pr_75255::bar[0]<alloc::string[0]::String[0]> @@ pr_75255-cgu.1[Internal]
//~ MONO_ITEM fn pr_75255::bar[0]<alloc::vec[0]::Vec[0]<alloc::string[0]::String[0]>> @@ pr_75255-cgu.1[Internal]
