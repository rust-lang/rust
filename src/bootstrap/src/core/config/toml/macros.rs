// We are using a decl macro instead of a derive proc macro here to reduce the compile time of bootstrap.
#[macro_export]
macro_rules! define_config {
    ($(#[$attr:meta])* struct $name:ident {
        $($field:ident: Option<$field_ty:ty> = $field_key:literal,)*
    }) => {
        $(#[$attr])*
        pub struct $name {
            $(pub $field: Option<$field_ty>,)*
        }

        impl Merge for $name {
            fn merge(
                &mut self,
                _parent_config_path: Option<PathBuf>,
                _included_extensions: &mut HashSet<PathBuf>,
                other: Self,
                replace: ReplaceOpt
            ) {
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
                                        exit!(2);
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
        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
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
                Deserializer::deserialize_struct(
                    deserializer,
                    stringify!($name),
                    FIELDS,
                    Field,
                )
            }
        }
    }
}

#[macro_export]
macro_rules! check_ci_llvm {
    ($name:expr) => {
        assert!(
            $name.is_none(),
            "setting {} is incompatible with download-ci-llvm.",
            stringify!($name).replace("_", "-")
        );
    };
}
