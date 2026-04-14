//! Service Contract Schema
//!
//! This module defines the canonical, machine-readable contract for every long-running
//! service in Thing-OS. The contract formalizes what the system already believes:
//! - Services are filesystem/device watchers, not boot-time scanners
//! - The VFS is the source of truth for resource discovery
//! - Services declare their dependencies and outputs explicitly
//!
//! ## Contract Fields
//!
//! - **Service Name**: Canonical name of the service
//! - **Watched Kinds**: Resource kinds this service watches (input)
//! - **Published Kinds**: Resource kinds this service publishes (output)
//! - **Published Properties**: Property keys this service sets
//! - **Idempotent**: Whether repeated operations produce the same result
//! - **Boot Assumptions**: MUST be empty for watch-driven services
//!
//! ## Runtime Enforcement
//!
//! Services MUST:
//! 1. Declare their contract before startup
//! 2. Register their contract entry at `/services/{name}`
//! 3. Only watch declared kinds
//! 4. Only publish declared kinds and properties
//!
//! ## Example Contract
//!
//! ```rust,ignore
//! use abi::service_contract::ServiceContract;
//!
//! const FLYTRAP_CONTRACT: ServiceContract = ServiceContract {
//!     name: "flytrap",
//!     watched_kinds: &["boot.Module", "content.Source"],
//!     published_kinds: &["Asset"],
//!     published_properties: &[
//!         "asset.name", "asset.kind", "asset.hash",
//!         "asset.size", "asset.bytespace", "asset.generation",
//!         "asset.source", "asset.ready"
//!     ],
//!     idempotent: true,
//!     boot_assumptions: &[],
//! };
//!
//! // At service startup:
//! fn main() {
//!     FLYTRAP_CONTRACT.validate().expect("Invalid contract");
//!     info!("flytrap contract validated");
//!     // ... continue with service initialization
//! }
//! ```

#![allow(dead_code)]

// Note: schema imports are reserved for future use in registration helpers
#[allow(unused_imports)]
use crate::schema::{keys, kinds, rels};

/// A service contract declaration
///
/// This struct defines the complete interface contract for a VFS-native service.
/// Services MUST declare their contract at startup and register it under `/services`.
#[derive(Debug, Clone)]
pub struct ServiceContract {
    /// Canonical service name (e.g., "flytrap", "blossom")
    pub name: &'static str,

    /// Resource kinds this service watches (input dependencies)
    ///
    /// Service MUST NOT watch kinds not declared here.
    /// Empty array means service doesn't watch any resources (clock-driven, etc.)
    pub watched_kinds: &'static [&'static str],

    /// Resource kinds this service publishes (output)
    ///
    /// Service MUST NOT create resources of kinds not declared here.
    /// Empty array means service doesn't create resources (pure transformer, etc.)
    pub published_kinds: &'static [&'static str],

    /// Property keys this service sets on published resources
    ///
    /// Service MUST NOT set properties not declared here on its published kinds.
    /// May also set properties on watched resources (transformations).
    pub published_properties: &'static [&'static str],

    /// Whether this service's operations are idempotent
    ///
    /// `true` means:
    /// - Processing the same input multiple times produces identical output
    /// - Service can be safely restarted without corrupting state
    /// - Watches can be replayed without side effects
    ///
    /// `false` means:
    /// - Service maintains internal state that cannot be reconstructed
    /// - Duplicate events may cause incorrect behavior
    /// - Special recovery procedures needed on restart
    pub idempotent: bool,

    /// Boot-time assumptions that MUST exist before service starts
    ///
    /// For watch-driven services, this MUST be empty!
    /// Non-empty values indicate a service that violates the watch-driven model.
    ///
    /// Example of INVALID assumptions (boot-shaped thinking):
    /// - "All fonts are loaded"
    /// - "Framebuffer exists"
    /// - "Network is available"
    ///
    /// Instead, services MUST watch for these resources and react when they appear.
    pub boot_assumptions: &'static [&'static str],
}

impl ServiceContract {
    /// Validate that a service contract is well-formed
    ///
    /// Returns `Ok(())` if the contract is valid, or an error string describing
    /// the violation.
    pub fn validate(&self) -> Result<(), &'static str> {
        // Service name must not be empty
        if self.name.is_empty() {
            return Err("Service name cannot be empty");
        }

        // Watch-driven services MUST NOT have boot assumptions
        if !self.boot_assumptions.is_empty() {
            return Err("Watch-driven services MUST NOT have boot assumptions");
        }

        // Service must either watch or publish (or both)
        if self.watched_kinds.is_empty() && self.published_kinds.is_empty() {
            return Err("Service must watch and/or publish resources");
        }

        Ok(())
    }

    /// Check if this service watches a given node kind
    pub fn watches_kind(&self, kind: &str) -> bool {
        self.watched_kinds.iter().any(|k| *k == kind)
    }

    /// Check if this service publishes a given node kind
    pub fn publishes_kind(&self, kind: &str) -> bool {
        self.published_kinds.iter().any(|k| *k == kind)
    }

    /// Check if this service publishes a given property
    pub fn publishes_property(&self, property: &str) -> bool {
        self.published_properties.iter().any(|p| *p == property)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_contract() {
        let contract = ServiceContract {
            name: "test_service",
            watched_kinds: &["boot.Module"],
            published_kinds: &["Asset"],
            published_properties: &["asset.name", "asset.hash"],
            idempotent: true,
            boot_assumptions: &[],
        };

        assert!(contract.validate().is_ok());
    }

    #[test]
    fn test_empty_name() {
        let contract = ServiceContract {
            name: "",
            watched_kinds: &["boot.Module"],
            published_kinds: &["Asset"],
            published_properties: &["asset.name"],
            idempotent: true,
            boot_assumptions: &[],
        };

        assert!(contract.validate().is_err());
    }

    #[test]
    fn test_boot_assumptions_forbidden() {
        let contract = ServiceContract {
            name: "bad_service",
            watched_kinds: &["boot.Module"],
            published_kinds: &["Asset"],
            published_properties: &["asset.name"],
            idempotent: true,
            boot_assumptions: &["Framebuffer exists"],
        };

        assert_eq!(
            contract.validate(),
            Err("Watch-driven services MUST NOT have boot assumptions")
        );
    }

    #[test]
    fn test_must_watch_or_publish() {
        let contract = ServiceContract {
            name: "useless_service",
            watched_kinds: &[],
            published_kinds: &[],
            published_properties: &[],
            idempotent: true,
            boot_assumptions: &[],
        };

        assert_eq!(
            contract.validate(),
            Err("Service must watch and/or publish resources")
        );
    }

    #[test]
    fn test_watches_kind() {
        let contract = ServiceContract {
            name: "test",
            watched_kinds: &["boot.Module", "Asset"],
            published_kinds: &[],
            published_properties: &[],
            idempotent: true,
            boot_assumptions: &[],
        };

        assert!(contract.watches_kind("boot.Module"));
        assert!(contract.watches_kind("Asset"));
        assert!(!contract.watches_kind("Unknown"));
    }

    #[test]
    fn test_publishes_kind() {
        let contract = ServiceContract {
            name: "test",
            watched_kinds: &[],
            published_kinds: &["Asset", "Font"],
            published_properties: &[],
            idempotent: true,
            boot_assumptions: &[],
        };

        assert!(contract.publishes_kind("Asset"));
        assert!(contract.publishes_kind("Font"));
        assert!(!contract.publishes_kind("Unknown"));
    }

    #[test]
    fn test_publishes_property() {
        let contract = ServiceContract {
            name: "test",
            watched_kinds: &[],
            published_kinds: &["Asset"],
            published_properties: &["asset.name", "asset.hash"],
            idempotent: true,
            boot_assumptions: &[],
        };

        assert!(contract.publishes_property("asset.name"));
        assert!(contract.publishes_property("asset.hash"));
        assert!(!contract.publishes_property("unknown.prop"));
    }
}
