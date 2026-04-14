# Service Contract System

## Overview

The Service Contract System formalizes and enforces the graph-native architecture in Thing-OS. It transforms the implicitly consistent watch-driven model into an explicit, machine-checkable system contract.

## Problem Statement

Previously, Thing-OS services were conceptually graph-native but lacked formal enforcement:
- Services implicitly followed watch-driven patterns
- No machine-readable declaration of service interfaces
- Boot assumptions could creep in undetected
- Service boundaries were socially enforced, not mechanically verified

This created drift risk: new services could easily violate the graph-native model.

## Solution

The Service Contract System provides:
1. **Mandatory contract declaration** for all services
2. **Machine-readable schema** for service interfaces
3. **Runtime validation** that contracts are well-formed
4. **Graph registration** of contracts at `/sys/services/{name}`
5. **Zero boot assumptions** requirement for graph-native services

## Contract Schema

Every service MUST declare a `ServiceContract` with:

```rust
pub struct ServiceContract {
    /// Canonical service name (e.g., "ingestd", "blossom")
    pub name: &'static str,
    
    /// Node kinds this service watches (input dependencies)
    pub watched_kinds: &'static [&'static str],
    
    /// Node kinds this service publishes (output)
    pub published_kinds: &'static [&'static str],
    
    /// Property keys this service sets on published nodes
    pub published_properties: &'static [&'static str],
    
    /// Whether this service's operations are idempotent
    pub idempotent: bool,
    
    /// Boot-time assumptions (MUST be empty for graph-native services!)
    pub boot_assumptions: &'static [&'static str],
}
```

## Contract Properties

Contracts are published to the graph with these properties:

- `service.contract.name`: Service canonical name
- `service.contract.watched_kinds`: Bytespace containing watched kind list
- `service.contract.published_kinds`: Bytespace containing published kind list
- `service.contract.published_properties`: Bytespace containing property key list
- `service.contract.idempotent`: 1 if idempotent, 0 if stateful
- `service.contract.boot_assumptions`: Bytespace (MUST be empty)
- `service.contract.status`: "declared", "registered", "active", "error"
- `service.contract.version`: Monotonic version counter

## Contract Validation

Contracts are validated at declaration time:

1. ✅ Service name MUST NOT be empty
2. ✅ Boot assumptions MUST be empty (graph-native requirement)
3. ✅ Service MUST watch and/or publish nodes
4. ✅ All kind names MUST be valid schema constants
5. ✅ All property keys MUST be valid schema constants

## Service Lifecycle

### 1. Declaration (Compile-Time)

```rust
use abi::service_contract::ServiceContract;

const INGESTD_CONTRACT: ServiceContract = ServiceContract {
    name: "ingestd",
    watched_kinds: &["boot.Module", "content.Source"],
    published_kinds: &["Asset"],
    published_properties: &[
        "asset.name",
        "asset.kind", 
        "asset.hash",
        "asset.size",
        "asset.bytespace",
        "asset.generation",
        "asset.source",
        "asset.ready",
    ],
    idempotent: true,
    boot_assumptions: &[], // MUST be empty!
};
```

### 2. Validation (Service Startup)

```rust
fn main() {
    // Validate contract is well-formed
    INGESTD_CONTRACT.validate()
        .expect("Invalid service contract");
    
    info!("Service contract validated: {}", INGESTD_CONTRACT.name);
    
    // Continue with service initialization...
}
```

### 3. Registration (Future Enhancement)

In the future, services will register their contracts in the graph:

```rust
// Create contract node at /sys/services/ingestd
let contract_id = register_service_contract(&INGESTD_CONTRACT)?;

// Link service instance to contract
link(service_id, rels::IMPLEMENTS_CONTRACT, contract_id)?;
```

## Example: Asset Watcher Service (ingestd)

```rust
const INGESTD_CONTRACT: ServiceContract = ServiceContract {
    name: "ingestd",
    
    // What we watch (inputs)
    watched_kinds: &[
        "boot.Module",      // Limine boot modules
        "content.Source",   // ISO, disk mounts, etc.
    ],
    
    // What we publish (outputs)
    published_kinds: &[
        "Asset",            // Canonical asset nodes
    ],
    
    // Properties we set
    published_properties: &[
        "asset.name",       // Asset filename/path
        "asset.kind",       // Type: font, svg, image, etc.
        "asset.hash",       // SHA-256 content hash
        "asset.size",       // Size in bytes
        "asset.bytespace",  // Content reference
        "asset.generation", // Change counter
        "asset.source",     // Origin: "boot", "iso9660", etc.
        "asset.ready",      // 1 when ready for use
    ],
    
    // Idempotency
    idempotent: true,   // Same inputs → same outputs
    
    // Boot assumptions (FORBIDDEN)
    boot_assumptions: &[], // We watch for everything!
};
```

## Benefits

### 1. Explicit Over Implicit
Contracts make service interfaces visible and queryable in the graph.

### 2. Mechanical Verification
Invalid contracts are caught at compile-time (const validation) or startup.

### 3. Enforced Graph-Native Architecture
Services CANNOT declare boot assumptions. The graph is the only truth.

### 4. Self-Documenting
Contract declarations serve as authoritative interface documentation.

### 5. Tooling Foundation
Contracts enable automated:
- Service discovery and introspection
- Dependency graph analysis
- Impact analysis for schema changes
- Service health monitoring

## Anti-Patterns (Forbidden)

### ❌ Boot Assumptions

```rust
// WRONG: This violates the graph-native model!
boot_assumptions: &[
    "Framebuffer exists",
    "All fonts loaded",
    "Network available",
]
```

Services MUST watch for these things and react when they appear.

### ❌ Undeclared Watchers

```rust
// WRONG: Service watches kinds not in contract!
watched_kinds: &["Asset"],

// But code watches:
watch_subscribe(KIND_FONT_FACE, ...)?; // Violation!
```

### ❌ Undeclared Publishers

```rust
// WRONG: Service publishes kinds not in contract!
published_kinds: &["Asset"],

// But code creates:
create_node("font.Face")?; // Violation!
```

## Future Enhancements

### Runtime Enforcement
- Check that services only watch declared kinds
- Check that services only publish declared kinds
- Verify property writes match declarations

### Graph Registration
- Automatic contract node creation at `/sys/services/{name}`
- Service instance → contract relationships
- Contract versioning and migration

### Tooling
- `thing-os-contracts` CLI for introspection
- Contract diff tool for breaking change detection
- Service dependency visualizer

### Metrics
- Track contract violations in telemetry
- Service health based on contract compliance
- Alert on undeclared behavior

## Migration Path

### Phase 1: Declaration (Current)
- Add contract declarations to existing services
- Validate contracts at startup
- No enforcement yet

### Phase 2: Registration (Next)
- Register contracts in graph at service startup
- Create `/sys/services/` hierarchy
- Link service instances to contracts

### Phase 3: Enforcement (Future)
- Runtime checks for watch/publish violations
- Reject operations outside contract scope
- Full mechanical verification

## Testing

Contract validation is tested in `abi/src/service_contract.rs`:

```bash
cargo test -p abi service_contract
```

## See Also

- `abi/src/service_contract.rs` - Contract schema implementation
- `abi/src/schema.rs` - Property and kind constants
- `ASSET_WATCHER_SERVICE.md` - Example graph-native service
- `docs/platform.md` - Platform layer contract
