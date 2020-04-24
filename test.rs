#![allow(missing_docs)]

use embedded_hal::digital::v2::{InputPin, OutputPin};
use generic_array::{ArrayLength, GenericArray};
use heapless::Vec;

pub trait HeterogenousArray {
    type Len;
}

/// Macro to implement a iterator on trait objects from a tuple struct.
#[macro_export]
macro_rules! impl_heterogenous_array {
    ($s:ident, $t:ty, $len:tt, [$($idx:tt),+]) => {
        impl<'a> IntoIterator for &'a $s {
            type Item = &'a $t;
            type IntoIter = generic_array::GenericArrayIter<&'a $t, $len>;
            fn into_iter(self) -> Self::IntoIter {
                self.as_array().into_iter()
            }
        }
        impl<'a> IntoIterator for &'a mut $s {
            type Item = &'a mut $t;
            type IntoIter = generic_array::GenericArrayIter<&'a mut $t, $len>;
            fn into_iter(self) -> Self::IntoIter {
                self.as_mut_array().into_iter()
            }
        }
        impl $crate::matrix::HeterogenousArray for $s {
            type Len = $len;
        }
        impl $s {
            pub fn as_array(&self) -> generic_array::GenericArray<&$t, $len> {
                generic_array::arr![&$t; $( &self.$idx as &$t, )+]
            }
            pub fn as_mut_array(&mut self) -> generic_array::GenericArray<&mut $t, $len> {
                generic_array::arr![&mut $t; $( &mut self.$idx as &mut $t, )+]
            }
        }
    }
}

pub struct Matrix<C, R> {
    cols: C,
    rows: R,
}

impl<C, R> Matrix<C, R> {
    pub fn new<E>(cols: C, rows: R) -> Result<Self, E>
        where
                for<'a> &'a mut R: IntoIterator<Item = &'a mut dyn OutputPin<Error = E>>,
    {
        let mut res = Self { cols, rows };
        res.clear()?;
        Ok(res)
    }
    pub fn clear<'a, E: 'a>(&'a mut self) -> Result<(), E>
        where
            &'a mut R: IntoIterator<Item = &'a mut dyn OutputPin<Error = E>>,
    {
        for r in self.rows.into_iter() {
            r.set_high()?;
        }
        Ok(())
    }
    pub fn get<'a, E: 'a>(&'a mut self) -> Result<PressedKeys<R::Len, C::Len>, E>
        where
            &'a mut R: IntoIterator<Item = &'a mut dyn OutputPin<Error = E>>,
            R: HeterogenousArray,
            R::Len: ArrayLength<GenericArray<bool, C::Len>>,
            &'a C: IntoIterator<Item = &'a dyn InputPin<Error = E>>,
            C: HeterogenousArray,
            C::Len: ArrayLength<bool>,
    {
        let cols = &self.cols;
        self.rows
            .into_iter()
            .map(|r| {
                r.set_low()?;
                let col = cols
                    .into_iter()
                    .map(|c| c.is_low())
                    .collect::<Result<Vec<_, C::Len>, E>>()?
                    .into_iter()
                    .collect();
                r.set_high()?;
                Ok(col)
            })
            .collect::<Result<Vec<_, R::Len>, E>>()
            .map(|res| PressedKeys(res.into_iter().collect()))
    }
}

#[derive(Default, PartialEq, Eq)]
pub struct PressedKeys<U, V>(pub GenericArray<GenericArray<bool, V>, U>)
    where
        V: ArrayLength<bool>,
        U: ArrayLength<GenericArray<bool, V>>;

impl<U, V> PressedKeys<U, V>
    where
        V: ArrayLength<bool>,
        U: ArrayLength<GenericArray<bool, V>>,
{
    pub fn iter_pressed<'a>(&'a self) -> impl Iterator<Item = (usize, usize)> + Clone + 'a {
        self.0.iter().enumerate().flat_map(|(i, r)| {
            r.iter()
                .enumerate()
                .filter_map(move |(j, &b)| if b { Some((i, j)) } else { None })
        })
    }
}

impl<'a, U, V> IntoIterator for &'a PressedKeys<U, V>
    where
        V: ArrayLength<bool>,
        U: ArrayLength<GenericArray<bool, V>>,
        U: ArrayLength<&'a GenericArray<bool, V>>,
{
    type IntoIter = core::slice::Iter<'a, GenericArray<bool, V>>;
    type Item = &'a GenericArray<bool, V>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}