//! Internationalization (i18n) support for Thing-OS.
//!
//! This module provides a first-class i18n system where every UI-visible string
//! is represented as a (key, fallback) pair, enabling runtime locale switching.
//!
//! ## Design
//!
//! - `TextKey`: Stable identifier for translatable strings
//! - `LocalizedText`: Combines a key with a fallback string
//! - `LocaleId`: Identifies a locale (e.g., en-US, la)
//! - `Catalog`: Maps text keys to translated strings for a locale
//! - `Translator`: Provides translation services with fallback chain
//!
//! ## Usage
//!
//! ```ignore
//! use stem::i18n::LocalizedText;
//!
//! // Define a text key with fallback using the t! macro
//! const WINDOW_TITLE: LocalizedText = stem::t!("ui.fonts.title", "Font Explorer");
//!
//! // Get translated string for current locale
//! let title = WINDOW_TITLE.get();
//! ```

use alloc::collections::BTreeMap;
use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use spin::Mutex;

/// A stable identifier for a translatable string.
///
/// Text keys are used to look up translations in locale catalogs.
/// They should be hierarchical and descriptive (e.g., "ui.fonts.title").
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TextKey(&'static str);

impl TextKey {
    /// Create a new text key from a static string.
    pub const fn new(key: &'static str) -> Self {
        Self(key)
    }

    /// Get the key string.
    pub const fn as_str(&self) -> &'static str {
        self.0
    }
}

/// A locale identifier (e.g., "en-US", "la", "syc").
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LocaleId(&'static str);

impl LocaleId {
    /// Create a new locale identifier.
    pub const fn new(id: &'static str) -> Self {
        Self(id)
    }

    /// Get the locale string.
    pub const fn as_str(&self) -> &'static str {
        self.0
    }

    /// English (US) locale
    pub const EN_US: LocaleId = LocaleId("en-US");

    /// Latin locale
    pub const LA: LocaleId = LocaleId("la");

    /// Syriac locale
    pub const SYC: LocaleId = LocaleId("syc");
}

/// A localizable text with a key and fallback string.
///
/// This is the primary type used throughout the UI code.
/// It combines a stable text key with a fallback string that is used
/// when no translation is available.
#[derive(Clone, Copy, Debug)]
pub struct LocalizedText {
    /// The translation key
    pub key: TextKey,
    /// The fallback text (typically English)
    pub fallback: &'static str,
}

impl LocalizedText {
    /// Create a new localized text.
    pub const fn new(key: &'static str, fallback: &'static str) -> Self {
        Self {
            key: TextKey::new(key),
            fallback,
        }
    }

    /// Get the translated string for the current locale.
    ///
    /// Falls back to the fallback string if no translation is available.
    pub fn get(&self) -> &str {
        TRANSLATOR.translate(self.key).unwrap_or(self.fallback)
    }

    /// Get the translation key.
    pub const fn key(&self) -> TextKey {
        self.key
    }

    /// Get the fallback string.
    pub const fn fallback(&self) -> &'static str {
        self.fallback
    }
}

/// Convenient macro for creating LocalizedText instances.
///
/// # Example
/// ```ignore
/// const TITLE: LocalizedText = t!("ui.window.title", "Window Title");
/// ```
#[macro_export]
macro_rules! t {
    ($key:expr, $fallback:expr) => {
        $crate::i18n::LocalizedText::new($key, $fallback)
    };
}

/// A translation catalog for a specific locale.
///
/// Maps text keys to translated strings.
pub struct Catalog {
    locale: LocaleId,
    translations: BTreeMap<&'static str, &'static str>,
}

impl Catalog {
    /// Create a new empty catalog for a locale.
    pub const fn new(locale: LocaleId) -> Self {
        Self {
            locale,
            translations: BTreeMap::new(),
        }
    }

    /// Create a catalog from a static translation table.
    pub fn from_table(locale: LocaleId, table: &[(&'static str, &'static str)]) -> Self {
        let mut catalog = Self::new(locale);
        for (key, translation) in table {
            catalog.translations.insert(key, translation);
        }
        catalog
    }

    /// Get a translation for a key.
    pub fn get(&self, key: TextKey) -> Option<&'static str> {
        self.translations.get(key.as_str()).copied()
    }

    /// Get the locale of this catalog.
    pub fn locale(&self) -> LocaleId {
        self.locale
    }
}

/// Global translator that manages locale switching and translation lookups.
pub struct Translator {
    current_locale: AtomicU32,
    /// Increments each time locale changes (for UI cache invalidation)
    generation: AtomicU64,
    catalogs: Mutex<BTreeMap<LocaleId, Catalog>>,
}

impl Translator {
    /// Create a new translator.
    const fn new() -> Self {
        Self {
            current_locale: AtomicU32::new(0), // 0 = EN_US
            generation: AtomicU64::new(0),
            catalogs: Mutex::new(BTreeMap::new()),
        }
    }

    /// Initialize the translator with default catalogs.
    pub fn init(&self) {
        let mut catalogs = self.catalogs.lock();

        // English catalog (mostly empty, uses fallbacks)
        catalogs.insert(LocaleId::EN_US, Catalog::from_table(LocaleId::EN_US, &[]));

        // Latin catalog
        catalogs.insert(
            LocaleId::LA,
            Catalog::from_table(LocaleId::LA, &LATIN_CATALOG),
        );

        // Syriac catalog (placeholder for future)
        catalogs.insert(LocaleId::SYC, Catalog::from_table(LocaleId::SYC, &[]));
    }

    /// Get the current locale.
    pub fn current_locale(&self) -> LocaleId {
        match self.current_locale.load(Ordering::Relaxed) {
            0 => LocaleId::EN_US,
            1 => LocaleId::LA,
            2 => LocaleId::SYC,
            _ => LocaleId::EN_US,
        }
    }

    /// Set the current locale.
    pub fn set_locale(&self, locale: LocaleId) {
        let index = if locale.as_str() == LocaleId::EN_US.as_str() {
            0
        } else if locale.as_str() == LocaleId::LA.as_str() {
            1
        } else if locale.as_str() == LocaleId::SYC.as_str() {
            2
        } else {
            // Unknown locale - default to EN_US
            0
        };

        if self.current_locale.load(Ordering::Relaxed) != index {
            self.current_locale.store(index, Ordering::Relaxed);
            self.generation.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Cycle to the next locale.
    pub fn cycle_locale(&self) {
        let current = self.current_locale.load(Ordering::Relaxed);
        let next = (current + 1) % 3; // 3 locales: EN_US, LA, SYC
        self.current_locale.store(next, Ordering::Relaxed);
        self.generation.fetch_add(1, Ordering::Relaxed);
    }

    /// Get the current generation counter.
    ///
    /// This increments each time the locale changes, allowing UIs to detect
    /// when they need to redraw with new translations.
    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Relaxed)
    }

    /// Translate a text key to the current locale.
    ///
    /// Returns None if no translation is found (caller should use fallback).
    pub fn translate(&self, key: TextKey) -> Option<&'static str> {
        let locale = self.current_locale();
        let catalogs = self.catalogs.lock();

        // Try current locale
        if let Some(catalog) = catalogs.get(&locale) {
            if let Some(translation) = catalog.get(key) {
                return Some(translation);
            }
        }

        // Try default locale (EN_US)
        if locale != LocaleId::EN_US {
            if let Some(catalog) = catalogs.get(&LocaleId::EN_US) {
                if let Some(translation) = catalog.get(key) {
                    return Some(translation);
                }
            }
        }

        None
    }
}

/// Global translator instance.
static TRANSLATOR: Translator = Translator::new();

/// Initialize the i18n system.
///
/// This should be called once at application startup.
pub fn init() {
    TRANSLATOR.init();
}

/// Get the current locale.
pub fn current_locale() -> LocaleId {
    TRANSLATOR.current_locale()
}

/// Set the current locale.
pub fn set_locale(locale: LocaleId) {
    TRANSLATOR.set_locale(locale);
}

/// Cycle to the next locale.
pub fn cycle_locale() {
    TRANSLATOR.cycle_locale();
}

/// Get the current i18n generation counter.
pub fn generation() -> u64 {
    TRANSLATOR.generation()
}

/// Translate a text key.
pub fn translate(key: TextKey) -> Option<&'static str> {
    TRANSLATOR.translate(key)
}

// ============================================================================
// Locale Catalogs
// ============================================================================

/// Latin translation catalog
const LATIN_CATALOG: &[(&str, &str)] = &[
    // Font Explorer
    ("ui.fonts.title", "Litterae"),
    ("ui.fonts.explorer", "Explorator Litterarum"),
    ("ui.fonts.count", "Litterae"),
    // Photosynthesis
    ("ui.photosynthesis.title", "Photosynthesis"),
    // Common UI
    ("ui.window.close", "Claudere"),
    ("ui.window.minimize", "Minuere"),
    ("ui.window.maximize", "Augere"),
    // Sample text for font display
    (
        "ui.fonts.sample",
        "Sphinx Iovis dura lex sed lex. 0123456789",
    ),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_key() {
        let key = TextKey::new("ui.test.key");
        assert_eq!(key.as_str(), "ui.test.key");
    }

    #[test]
    fn test_locale_id() {
        assert_eq!(LocaleId::EN_US.as_str(), "en-US");
        assert_eq!(LocaleId::LA.as_str(), "la");
        assert_eq!(LocaleId::SYC.as_str(), "syc");
    }

    #[test]
    fn test_localized_text() {
        let text = LocalizedText::new("test.key", "Fallback Text");
        assert_eq!(text.key().as_str(), "test.key");
        assert_eq!(text.fallback(), "Fallback Text");
    }

    #[test]
    fn test_catalog() {
        let table = [("key1", "Translation 1"), ("key2", "Translation 2")];
        let catalog = Catalog::from_table(LocaleId::LA, &table);

        assert_eq!(catalog.get(TextKey::new("key1")), Some("Translation 1"));
        assert_eq!(catalog.get(TextKey::new("key2")), Some("Translation 2"));
        assert_eq!(catalog.get(TextKey::new("key3")), None);
    }

    #[test]
    fn test_translator_locale_switching() {
        let translator = Translator::new();
        translator.init();

        assert_eq!(translator.current_locale(), LocaleId::EN_US);

        translator.set_locale(LocaleId::LA);
        assert_eq!(translator.current_locale(), LocaleId::LA);

        translator.set_locale(LocaleId::SYC);
        assert_eq!(translator.current_locale(), LocaleId::SYC);
    }

    #[test]
    fn test_translator_cycle() {
        let translator = Translator::new();
        translator.init();

        assert_eq!(translator.current_locale(), LocaleId::EN_US);

        translator.cycle_locale();
        assert_eq!(translator.current_locale(), LocaleId::LA);

        translator.cycle_locale();
        assert_eq!(translator.current_locale(), LocaleId::SYC);

        translator.cycle_locale();
        assert_eq!(translator.current_locale(), LocaleId::EN_US);
    }

    #[test]
    fn test_translator_generation() {
        let translator = Translator::new();
        translator.init();

        let gen1 = translator.generation();
        translator.set_locale(LocaleId::LA);
        let gen2 = translator.generation();

        assert!(gen2 > gen1);
    }

    #[test]
    fn test_translation() {
        let translator = Translator::new();
        translator.init();

        // English (no translation, returns None)
        translator.set_locale(LocaleId::EN_US);
        assert_eq!(translator.translate(TextKey::new("ui.fonts.title")), None);

        // Latin (has translation)
        translator.set_locale(LocaleId::LA);
        assert_eq!(
            translator.translate(TextKey::new("ui.fonts.title")),
            Some("Litterae")
        );
    }
}
