//! This module defines two macros:
//!
//! - `define_config!`: A declarative macro used instead of `#[derive(Deserialize)]` to reduce
//!   compile time and binary size, especially for the bootstrap binary.
//!
//! - `check_ci_llvm!`: A compile-time assertion macro that ensures certain settings are
//!   not enabled when `download-ci-llvm` is active.
//!
//! A declarative macro is used here in place of a procedural derive macro to minimize
//! the compile time of the bootstrap process.
//!

// We are using a decl macro instead of a derive proc macro here to reduce the compile time of bootstrap.
macro_rules! define_config {
    (
        $(#[$attr:meta])*
            struct $name:ident {
            $(
                $(#[$field_attr:meta])*
                $field:ident: Option<$field_ty:ty> = $field_key:literal,
            )*
        }
    ) => {
        $(#[$attr])*
        pub struct $name {
            $(
                $(#[$field_attr])*
                pub $field: Option<$field_ty>,
            )*
        }

        impl crate::core::config::Merge for $name {
            fn merge(
                &mut self,
                _parent_config_path: Option<std::path::PathBuf>,
                _included_extensions: &mut std::collections::HashSet<std::path::PathBuf>,
                other: Self,
                replace: crate::core::config::ReplaceOpt
            ) {
                use crate::core::config::ReplaceOpt;
                $(
                    match replace {
                        ReplaceOpt::IgnoreDuplicate => {
                            if self.$field.is_none() {
                                self.$field = other.$field;
                            }
                        },
                        ReplaceOpt::Override => {
                            if other.$field.is_some() {
                                self.$field = other.$field;
                            }
                        }
                        ReplaceOpt::ErrorOnDuplicate => {
                            if other.$field.is_some() {
                                if self.$field.is_some() {
                                    if cfg!(test) {
                                        panic!("overriding existing option")
                                    } else {
                                        eprintln!("overriding existing option: `{}`", stringify!($field));
                                        $crate::utils::helpers::exit_process(2);
                                    }
                                } else {
                                    self.$field = other.$field;
                                }
                            }
                        }
                    }
                )*
            }
        }

        // The following is a trimmed version of what serde_derive generates. All parts not relevant
        // for toml deserialization have been removed. This reduces the binary size and improves
        // compile time of bootstrap.
        impl<'de> serde::Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct Field;
                impl<'de> serde::de::Visitor<'de> for Field {
                    type Value = $name;
                    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        f.write_str(concat!("struct ", stringify!($name)))
                    }

                    #[inline]
                    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
                    where
                        A: serde::de::MapAccess<'de>,
                    {
                        $(let mut $field: Option<$field_ty> = None;)*
                        while let Some(key) =
                            match serde::de::MapAccess::next_key::<String>(&mut map) {
                                Ok(val) => val,
                                Err(err) => {
                                    return Err(err);
                                }
                            }
                        {
                            match &*key {
                                $($field_key => {
                                    if $field.is_some() {
                                        return Err(<A::Error as serde::de::Error>::duplicate_field(
                                            $field_key,
                                        ));
                                    }
                                    $field = match serde::de::MapAccess::next_value::<$field_ty>(
                                        &mut map,
                                    ) {
                                        Ok(val) => Some(val),
                                        Err(err) => {
                                            return Err(err);
                                        }
                                    };
                                })*
                                key => {
                                    return Err(serde::de::Error::unknown_field(key, FIELDS));
                                }
                            }
                        }
                        Ok($name { $($field),* })
                    }
                }
                const FIELDS: &'static [&'static str] = &[
                    $($field_key,)*
                ];
                serde::Deserializer::deserialize_struct(
                    deserializer,
                    stringify!($name),
                    FIELDS,
                    Field,
                )
            }
        }
    }
}

macro_rules! check_ci_llvm {
    ($name:expr) => {
        assert!(
            $name.is_none(),
            "setting {} is incompatible with download-ci-llvm.",
            stringify!($name).replace("_", "-")
        );
    };
}

pub(crate) use check_ci_llvm;
pub(crate) use define_config;
