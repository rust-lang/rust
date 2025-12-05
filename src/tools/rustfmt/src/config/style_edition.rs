use crate::config::StyleEdition;

/// Defines the default value for the given style edition
#[allow(dead_code)]
pub trait StyleEditionDefault {
    type ConfigType;
    fn style_edition_default(style_edition: StyleEdition) -> Self::ConfigType;
}

/// macro to help implement `StyleEditionDefault` for config options
#[macro_export]
macro_rules! style_edition_default {
    ($ty:ident, $config_ty:ty, _ => $default:expr) => {
        impl $crate::config::style_edition::StyleEditionDefault for $ty {
            type ConfigType = $config_ty;

            fn style_edition_default(_: $crate::config::StyleEdition) -> Self::ConfigType {
                $default
            }
        }
    };
    ($ty:ident, $config_ty:ty, Edition2024 => $default_2024:expr, _ => $default_2015:expr) => {
        impl $crate::config::style_edition::StyleEditionDefault for $ty {
            type ConfigType = $config_ty;

            fn style_edition_default(
                style_edition: $crate::config::StyleEdition,
            ) -> Self::ConfigType {
                match style_edition {
                    $crate::config::StyleEdition::Edition2015
                    | $crate::config::StyleEdition::Edition2018
                    | $crate::config::StyleEdition::Edition2021 => $default_2015,
                    $crate::config::StyleEdition::Edition2024 => $default_2024,
                }
            }
        }
    };
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::config::StyleEdition;

    #[test]
    fn test_impl_default_style_edition_struct_for_all_editions() {
        struct Unit;
        style_edition_default!(Unit, usize, _ => 100);

        // regardless of the style edition used the value will always return 100
        assert_eq!(Unit::style_edition_default(StyleEdition::Edition2015), 100);
        assert_eq!(Unit::style_edition_default(StyleEdition::Edition2018), 100);
        assert_eq!(Unit::style_edition_default(StyleEdition::Edition2021), 100);
        assert_eq!(Unit::style_edition_default(StyleEdition::Edition2024), 100);
    }

    #[test]
    fn test_impl_default_style_edition_for_old_and_new_editions() {
        struct Unit;
        style_edition_default!(Unit, usize, Edition2024 => 50, _ => 100);

        // style edition 2015-2021 are all the same
        assert_eq!(Unit::style_edition_default(StyleEdition::Edition2015), 100);
        assert_eq!(Unit::style_edition_default(StyleEdition::Edition2018), 100);
        assert_eq!(Unit::style_edition_default(StyleEdition::Edition2021), 100);

        // style edition 2024
        assert_eq!(Unit::style_edition_default(StyleEdition::Edition2024), 50);
    }
}
