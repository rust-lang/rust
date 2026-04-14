//! Shared schema constants for Unified Device Graph v0.2

pub mod source {
    pub const BOOT: u8 = 0;
    pub const PLATFORM: u8 = 1;
    pub const DTB: u8 = 2;
    pub const ACPI: u8 = 3;
    pub const PCI: u8 = 4;
    pub const PROBE: u8 = 5;
}

pub mod confidence {
    pub const LOW: u8 = 0;
    pub const MEDIUM: u8 = 1;
    pub const HIGH: u8 = 2;
}

/// Content source kind constants
pub mod content_source_kind {
    /// Limine boot modules
    pub const LIMINE_MODULE: &str = "limine_module";
    /// ISO9660 filesystem on disk
    pub const ISO9660_DISK: &str = "iso9660_disk";
}

/// Content source state constants
pub mod content_source_state {
    /// Source is ready and serving content
    pub const READY: &str = "ready";
    /// Source encountered an error
    pub const ERROR: &str = "error";
    /// Source is initializing
    pub const INITIALIZING: &str = "initializing";
}

pub mod keys {
    pub const SOURCE: &str = "source";
    pub const CONFIDENCE: &str = "confidence";
    pub const NAME: &str = "name";
    pub const PHYS_BASE: &str = "phys_base";
    pub const SIZE_BYTES: &str = "size_bytes";
    pub const FORMAT: &str = "format";
    pub const HHDM_OFFSET: &str = "hhdm_offset";
    pub const START: &str = "start";
    pub const END: &str = "end";
    pub const WIDTH: &str = "width";
    pub const HEIGHT: &str = "height";
    pub const STRIDE: &str = "stride";
    pub const BPP: &str = "bpp";
    pub const BYTESPACE: &str = "bytespace"; // Used for Sprout shortcut
    pub const IRQ: &str = "irq";
    pub const UNIX_SECONDS: &str = "unix_seconds";
    pub const OFFSET_NS: &str = "offset_ns";
    pub const STATE: &str = "state";
    pub const QUALITY: &str = "quality";

    pub const LAST_UPDATED_MONO_NS: &str = "last_updated_mono_ns";

    // XML Properties
    pub const TAG: &str = "tag";
    pub const TEXT: &str = "text";
    pub const ATTR_NAME: &str = "attr_name";
    pub const ATTR_VALUE: &str = "attr_value";
    /// Stable sibling/attribute ordering (u32, monotonic within parent)
    pub const XML_ORDER: &str = "xml.order";

    // HTML Properties (reuses XML properties + additions)
    /// Child ordering in HTML document (u32, monotonic within parent)
    pub const HTML_ORDER: &str = "html.order";
    /// HTML element tag name (interned)
    pub const HTML_TAG: &str = "html.tag";

    // CSS Properties
    /// CSS property name (e.g., "color", "margin")
    pub const CSS_PROPERTY: &str = "css.property";
    /// CSS property value (e.g., "red", "10px")
    pub const CSS_VALUE: &str = "css.value";
    /// Raw CSS selector text
    pub const CSS_SELECTOR_TEXT: &str = "css.selector_text";
    /// CSS rule order within stylesheet
    pub const CSS_ORDER: &str = "css.order";
    /// CSS at-rule name (e.g., "media", "keyframes")
    pub const CSS_AT_RULE: &str = "css.at_rule";
    /// CSS at-rule prelude/condition text
    pub const CSS_PRELUDE: &str = "css.prelude";

    // Legacy mapping (to be deprecated or mapped)
    pub const KIND: &str = "kind";

    pub const VENDOR_ID: &str = "vendor_id";
    pub const DEVICE_ID: &str = "device_id";
    pub const VENDOR_NAME: &str = "vendor_name";
    pub const DEVICE_NAME: &str = "device_name";
    pub const PCI_NAME_SOURCE: &str = "pci_name_source";
    pub const CLASS_CODE: &str = "class_code";
    pub const SUBCLASS_CODE: &str = "subclass_code";
    pub const PROG_IF: &str = "prog_if";
    pub const REVISION_ID: &str = "revision_id";
    pub const BIND_KIND: &str = "bind_kind";
    pub const BIND_HASH: &str = "bind_hash";
    pub const BUS: &str = "bus";
    pub const DEVICE: &str = "device";
    pub const FUNCTION: &str = "function";
    pub const BAR0: &str = "bar0";
    pub const BAR1: &str = "bar1";
    pub const BAR2: &str = "bar2";
    pub const BAR3: &str = "bar3";
    pub const BAR4: &str = "bar4";
    pub const BAR5: &str = "bar5";
    pub const PORT_START: &str = "port_start";
    pub const PORT_END: &str = "port_end";
    pub const IRQ_MODE: &str = "irq_mode";
    pub const VECTOR: &str = "vector";
    pub const MSI_CAPABLE: &str = "msi_capable";
    pub const MSIX_CAPABLE: &str = "msix_capable";

    // VirtIO PCI Capability Offsets
    pub const VIRTIO_COMMON_BAR: &str = "virtio.common_bar";
    pub const VIRTIO_COMMON_OFFSET: &str = "virtio.common_offset";
    pub const VIRTIO_NOTIFY_BAR: &str = "virtio.notify_bar";
    pub const VIRTIO_NOTIFY_OFFSET: &str = "virtio.notify_offset";
    pub const VIRTIO_NOTIFY_MULTIPLIER: &str = "virtio.notify_multiplier";
    pub const VIRTIO_ISR_BAR: &str = "virtio.isr_bar";
    pub const VIRTIO_ISR_OFFSET: &str = "virtio.isr_offset";
    pub const VIRTIO_DEVICE_BAR: &str = "virtio.device_bar";
    pub const VIRTIO_DEVICE_OFFSET: &str = "virtio.device_offset";

    // Storage
    pub const SECTOR_SIZE: &str = "sector_size";
    pub const SECTOR_COUNT: &str = "sector_count";
    pub const LBA48: &str = "lba48";
    pub const ATA_CHANNEL: &str = "ata_channel";
    pub const ATA_DRIVE: &str = "ata_drive";
    pub const MODEL: &str = "model";
    pub const SERIAL: &str = "serial";
    pub const READ_PORT_HANDLE: &str = "read_port_handle";
    pub const WRITE_PORT_HANDLE: &str = "write_port_handle";

    // Network
    pub const MAC_ADDRESS: &str = "mac_address";
    pub const LINK_STATUS: &str = "link_status";
    pub const MTU: &str = "mtu";
    pub const RX_PACKETS: &str = "rx_packets";
    pub const TX_PACKETS: &str = "tx_packets";
    pub const RX_BYTES: &str = "rx_bytes";
    pub const TX_BYTES: &str = "tx_bytes";
    pub const NET_HOST_NAME: &str = "net.host.name";
    pub const NET_HOST_LAST_SEEN: &str = "net.host.last_seen";
    pub const NET_HOST_EXPIRES_AT: &str = "net.host.expires_at";
    pub const NET_ADDR_FAMILY: &str = "net.addr.family";
    pub const NET_ADDR_IP: &str = "net.addr.ip";
    pub const NET_ADDR_LAST_SEEN: &str = "net.addr.last_seen";
    pub const NET_ADDR_EXPIRES_AT: &str = "net.addr.expires_at";
    pub const NET_SVC_TYPE_NAME: &str = "net.svc_type.name";
    pub const NET_SVC_TYPE_DOMAIN: &str = "net.svc_type.domain";
    pub const NET_INSTANCE_NAME: &str = "net.instance.name";
    pub const NET_INSTANCE_FQDN: &str = "net.instance.fqdn";
    pub const NET_INSTANCE_LAST_SEEN: &str = "net.instance.last_seen";
    pub const NET_INSTANCE_EXPIRES_AT: &str = "net.instance.expires_at";
    pub const NET_ENDPOINT_PORT: &str = "net.endpoint.port";
    pub const NET_ENDPOINT_PROTO: &str = "net.endpoint.proto";
    pub const NET_ENDPOINT_IP: &str = "net.endpoint.ip";
    pub const NET_ENDPOINT_PRIORITY: &str = "net.endpoint.priority";
    pub const NET_ENDPOINT_WEIGHT: &str = "net.endpoint.weight";
    pub const NET_SOCK_KEY: &str = "net.sock.key";
    pub const NET_SOCK_PROTO: &str = "net.sock.proto";
    pub const NET_SOCK_STATE: &str = "net.sock.state";
    pub const NET_SOCK_FD: &str = "net.sock.fd";
    pub const NET_SOCK_PID: &str = "net.sock.pid";
    pub const NET_SOCK_CREATED_AT: &str = "net.sock.created_at";
    pub const NET_SOCK_CLOSED_AT: &str = "net.sock.closed_at";
    pub const NET_CONN_KEY: &str = "net.conn.key";
    pub const NET_CONN_STATE: &str = "net.conn.state";
    pub const NET_CONN_BYTES_TX: &str = "net.conn.bytes_tx";
    pub const NET_CONN_BYTES_RX: &str = "net.conn.bytes_rx";
    pub const NET_CONN_PACKETS_TX: &str = "net.conn.packets_tx";
    pub const NET_CONN_PACKETS_RX: &str = "net.conn.packets_rx";
    pub const NET_CONN_LAST_SEEN: &str = "net.conn.last_seen";
    pub const NET_CONN_LAST_ERROR: &str = "net.conn.last_error";
    pub const NET_CONN_RTT_MS: &str = "net.conn.rtt_ms";
    pub const NET_TXT_HASH: &str = "net.txt.hash";
    pub const NET_LAST_SEEN: &str = "net.last_seen";
    pub const NET_EXPIRES_AT: &str = "net.expires_at";
    pub const NET_STALE: &str = "net.stale";
    pub const NET_ID_KEY: &str = "net.id_key";
    pub const NET_NECTAR_PUBLISH_DISCOVERY_TO_GRAPH: &str = "net.nectar.publish_discovery_to_graph";
    pub const NET_ADVERTISE_ROOT_INSTANCE: &str = "net.advertise_root.instance";
    pub const NET_DESIRED_SERVICE_TYPE: &str = "net.desired.service_type";
    pub const NET_DESIRED_INSTANCE_NAME: &str = "net.desired.instance_name";
    pub const NET_DESIRED_DOMAIN: &str = "net.desired.domain";
    pub const NET_DESIRED_PORT: &str = "net.desired.port";
    pub const NET_DESIRED_PROTO: &str = "net.desired.proto";
    pub const NET_DESIRED_HOSTNAME: &str = "net.desired.hostname";
    pub const NET_DESIRED_TXT: &str = "net.desired.txt";

    // Audio status
    pub const SOUND_BUFFERED_FRAMES: &str = "sound.buffered_frames";
    pub const SOUND_FREE_FRAMES: &str = "sound.free_frames";
    pub const SOUND_UNDERRUNS: &str = "sound.underruns";

    // UI Properties
    pub const UI_X: &str = "ui.x";
    pub const UI_Y: &str = "ui.y";
    pub const UI_WIDTH: &str = "ui.width";
    pub const UI_HEIGHT: &str = "ui.height";
    pub const UI_COLOR: &str = "ui.color";
    pub const UI_TEXT: &str = "ui.text";
    pub const UI_FONT: &str = "ui.font";
    pub const UI_FONT_SIZE: &str = "ui.font_size";
    pub const UI_RADIUS: &str = "ui.radius";
    pub const UI_TITLE: &str = "ui.title";
    pub const UI_WINDOW_ICON: &str = "ui.window.icon";
    /// Bytespace id of icon pixel/SVG data for list items
    pub const UI_ICON_BYTESPACE: &str = "ui.icon.bytespace";
    /// Icon color for list items (ARGB u32, used when no bytespace icon)
    pub const UI_ICON_COLOR: &str = "ui.icon.color";
    pub const UI_WINDOW_SHADED: &str = "ui.window.shaded";
    pub const UI_HIDDEN: &str = "ui.hidden";
    pub const UI_Z_INDEX: &str = "ui.z_index";
    pub const UI_KIND: &str = "ui.kind";
    pub const UI_VISIBLE: &str = "ui.visible";

    // Graph-UI layout/styling properties (used by UiTreeBuilder + Blossom graph_ui)
    /// Flex gap in pixels between children (u64)
    pub const UI_GAP: &str = "ui.gap";
    /// Padding in pixels (uniform, u64)
    pub const UI_PADDING: &str = "ui.padding";
    /// Flex align-items: 0=Start, 1=Center, 2=End, 3=Stretch
    pub const UI_ALIGN: &str = "ui.align";
    /// Flex justify-content: 0=Start, 1=Center, 2=End, 3=SpaceBetween
    pub const UI_JUSTIFY: &str = "ui.justify";
    /// Font name bytespace (string, e.g. "NotoSans-Regular")
    pub const UI_FONT_NAME: &str = "ui.font_name";
    /// Placeholder text bytespace for text input
    pub const UI_PLACEHOLDER: &str = "ui.placeholder";
    /// Placeholder text bytespace for text input (canonical alias)
    pub const UI_PLACEHOLDER_TEXT: &str = "ui.placeholder_text";
    /// Text input cursor position (u64)
    pub const UI_CURSOR_POS: &str = "ui.cursor_pos";
    /// Text input cursor position (u64, canonical alias)
    pub const UI_CURSOR: &str = "ui.cursor";
    /// Text input current value bytespace
    pub const UI_INPUT_VALUE: &str = "ui.input_value";
    /// Stable UI identity key (bytespace UTF-8 string)
    pub const UI_KEY: &str = "ui.key";
    /// Optional class list (bytespace UTF-8 string, whitespace-separated)
    pub const UI_CLASS: &str = "ui.class";
    /// Optional stylesheet attached to a window (ThingId as u64)
    pub const UI_STYLESHEET: &str = "ui.stylesheet";
    /// Global default stylesheet (ThingId as u64), typically set on ui.Crown
    pub const UI_STYLESHEET_DEFAULT: &str = "ui.stylesheet.default";
    /// Optional UI role/classification string for tooling (bytespace UTF-8 string)
    pub const UI_ROLE: &str = "ui.role";
    /// Whether a node can receive input focus (0/1)
    pub const UI_FOCUSABLE: &str = "ui.focusable";
    /// Optional text selection start byte offset
    pub const UI_SELECTION_START: &str = "ui.selection.start";
    /// Optional text selection end byte offset
    pub const UI_SELECTION_END: &str = "ui.selection.end";
    pub const UI_ENABLED: &str = "ui.enabled";
    /// Whether a list item is currently selected (0/1)
    pub const UI_SELECTED: &str = "ui.selected";
    pub const UI_FOCUSED: &str = "ui.focused";
    pub const UI_RANK: &str = "ui.rank";
    pub const UI_FIXED: &str = "ui.fixed";
    pub const UI_MANUAL_POSITION: &str = "ui.manual_position";

    // Graph layout properties (Photosynthesis)
    pub const LAYOUT_POS_X: &str = "layout.pos.x";
    pub const LAYOUT_POS_Y: &str = "layout.pos.y";
    pub const LAYOUT_SIZE_W: &str = "layout.size.w";
    pub const LAYOUT_SIZE_H: &str = "layout.size.h";
    pub const LAYOUT_VEL_X: &str = "layout.vel.x";
    pub const LAYOUT_VEL_Y: &str = "layout.vel.y";
    pub const LAYOUT_PIN: &str = "layout.pin";
    pub const LAYOUT_GEN: &str = "layout.gen";

    // User-authored layout properties (Graph UI, shared with Photosynthesis)
    /// UI layout X coordinate (f32 bits stored as u64)
    pub const UI_LAYOUT_X: &str = "ui.layout.x";
    /// UI layout Y coordinate (f32 bits stored as u64)
    pub const UI_LAYOUT_Y: &str = "ui.layout.y";
    /// UI layout Z coordinate (f32 bits stored as u64, reserved for 3D)
    pub const UI_LAYOUT_Z: &str = "ui.layout.z";
    /// Layout space identifier (interned, e.g. "graph_ui_v1", "photosynthesis")
    pub const UI_LAYOUT_SPACE: &str = "ui.layout.space";
    /// Layout generation counter for debugging/versioning
    pub const UI_LAYOUT_GEN: &str = "ui.layout.gen";

    /// Compositor has taken control of framebuffer (boot console should stop)
    pub const UI_COMPOSITOR_ACTIVE: &str = "ui.compositor.active";
    pub const UI_BUTTON_LABEL: &str = "ui.button.label";
    pub const UI_BUTTON_ACTION_ID: &str = "ui.button.action_id";
    pub const UI_BUTTON_PRESSED: &str = "ui.button.pressed";
    pub const UI_CHECKBOX_LABEL: &str = "ui.checkbox.label";
    pub const UI_CHECKBOX_CHECKED: &str = "ui.checkbox.checked";
    pub const UI_CHECKBOX_INDETERMINATE: &str = "ui.checkbox.indeterminate";
    pub const UI_CHECKBOX_VALUE_ID: &str = "ui.checkbox.value_id";
    /// Deprecated single-slot queue bytespace (superseded by UI_EVENT_LOG/UI_EVENT_CURSOR).
    pub const UI_EVENT_QUEUE: &str = "ui.event.queue";
    /// Append-only UI event stream bytespace.
    pub const UI_EVENT_LOG: &str = "ui.event.log";
    /// Consumer cursor into UI_EVENT_LOG (byte offset).
    pub const UI_EVENT_CURSOR: &str = "ui.event.cursor";
    pub const UI_EVENT_GEN: &str = "ui.event.gen";
    // DrawList (VFS-native vector program) properties
    /// Bytespace id containing packed drawlist commands.
    pub const UI_DRAWLIST_BYTESPACE: &str = "ui.drawlist.bytespace";
    /// Monotonic generation for drawlist updates.
    pub const UI_DRAWLIST_GEN: &str = "ui.drawlist.gen";
    /// Optional: ThingId owner (window/surface/widget subtree).
    pub const UI_DRAWLIST_OWNER: &str = "ui.drawlist.owner";
    /// Optional: Bounding rectangle bytespace (RectI32Wire).
    pub const UI_DRAWLIST_BOUNDS: &str = "ui.drawlist.bounds";
    /// Optional: Debug name string bytespace.
    pub const UI_DRAWLIST_DEBUG_NAME: &str = "ui.drawlist.debug_name";
    /// Monotonic generation for scene updates.
    pub const UI_SCENE_GEN: &str = "ui.scene.gen";
    /// Bytespace id containing packed paint ops.
    pub const UI_PAINT_BYTESPACE: &str = "ui.paint.bytespace";
    /// Monotonic generation for paint updates.
    pub const UI_PAINT_GEN: &str = "ui.paint.gen";
    /// Optional viewport bounds in pixels (RectI32Wire bytespace).
    pub const UI_VIEWPORT_BYTESPACE: &str = "ui.viewport.bytespace";
    /// Optional affine transform from window-local to device coords (Mat3x2fWire bytespace).
    pub const UI_TRANSFORM_BYTESPACE: &str = "ui.transform.bytespace";
    /// Optional clip program bytespace (packed clip paths/rect stack).
    pub const UI_CLIP_BYTESPACE: &str = "ui.clip.bytespace";
    /// Optional monotonic generation for clip updates.
    pub const UI_CLIP_GEN: &str = "ui.clip.gen";
    /// Optional monotonic generation for transform updates.
    pub const UI_TRANSFORM_GEN: &str = "ui.transform.gen";

    // Render cache roots (renderer-owned)
    /// Root node for renderer cache artifacts for a window.
    pub const RENDER_CACHE_ROOT: &str = "render.cache_root";
    /// Derived artifact hash for render nodes.
    pub const RENDER_HASH: &str = "render.hash";
    /// Schema version for render artifact hashing.
    pub const RENDER_SCHEMA_VERSION: &str = "render.schema_version";

    // Render debug stats
    pub const RENDER_DEBUG_PATHS: &str = "render.debug.paths";
    pub const RENDER_DEBUG_FLATTENED: &str = "render.debug.flattened";
    pub const RENDER_DEBUG_EDGES: &str = "render.debug.edges";
    pub const RENDER_DEBUG_TILES: &str = "render.debug.tiles";
    pub const RENDER_DEBUG_HIT: &str = "render.debug.cache_hits";
    pub const RENDER_DEBUG_MISS: &str = "render.debug.cache_misses";

    // Clock & Binding Properties
    pub const CLOCK_NOW_TEXT: &str = "clock.now_text";
    pub const CLOCK_TICK: &str = "clock.tick";
    pub const BINDING_SOURCE: &str = "binding.source";
    pub const BINDING_TARGET: &str = "binding.target";
    pub const BINDING_MAP: &str = "binding.map";
    pub const BINDING_TO: &str = "binding.to";
    pub const EDGE_KIND: &str = "edge.kind";
    pub const EDGE_WEIGHT: &str = "edge.weight";

    // UI Inline
    pub const UI_INLINE_MODE: &str = "ui.inline.mode";
    pub const UI_SVG_BYTES: &str = "ui.svg_bytes";

    // UI Layout & Style
    pub const UI_LAYOUT_MODE: &str = "ui.layout.mode";
    pub const UI_CENTER_X: &str = "ui.layout.center_x";
    pub const UI_CENTER_Y: &str = "ui.layout.center_y";
    pub const UI_FILL_PARENT: &str = "ui.layout.fill_parent";
    pub const UI_INSET_RIGHT: &str = "ui.layout.inset_right";
    pub const UI_INSET_BOTTOM: &str = "ui.layout.inset_bottom";
    pub const UI_INSET_LEFT: &str = "ui.layout.inset_left";
    pub const UI_INSET_TOP: &str = "ui.layout.inset_top";
    pub const UI_SCROLL_X: &str = "ui.scroll.x";
    pub const UI_SCROLL_Y: &str = "ui.scroll.y";
    pub const UI_CLIP: &str = "ui.clip";
    pub const UI_BG_COLOR: &str = "ui.style.bg_color";
    pub const UI_FG_COLOR: &str = "ui.style.fg_color";
    pub const UI_FONT_SIZE_PX: &str = "ui.style.font_size_px";
    // Declarative style rule selector fields.
    pub const UI_STYLE_MATCH_KIND: &str = "ui.style.match.kind";
    pub const UI_STYLE_MATCH_CLASS: &str = "ui.style.match.class";
    pub const UI_STYLE_MATCH_KEY: &str = "ui.style.match.key";
    pub const UI_STYLE_MATCH_FOCUSED: &str = "ui.style.match.focused";
    // Declarative style rule value fields.
    pub const UI_STYLE_COLOR: &str = "ui.style.color";
    pub const UI_STYLE_BACKGROUND: &str = "ui.style.background";
    pub const UI_STYLE_FONT_NAME: &str = "ui.style.font.name";
    pub const UI_STYLE_FONT_SIZE: &str = "ui.style.font.size";
    pub const UI_STYLE_PADDING: &str = "ui.style.padding";
    pub const UI_STYLE_GAP: &str = "ui.style.gap";
    pub const UI_STYLE_BORDER_WIDTH: &str = "ui.style.border.width";
    pub const UI_STYLE_BORDER_COLOR: &str = "ui.style.border.color";
    pub const UI_STYLE_MIN_WIDTH: &str = "ui.style.min_width";
    pub const UI_STYLE_MIN_HEIGHT: &str = "ui.style.min_height";
    pub const UI_STYLE_CURSOR_COLOR: &str = "ui.style.cursor.color";
    /// Snapshot bytespace id containing a view's latest presented pixels.
    ///
    /// Reserved for presenter-owned updates (Blossom).
    pub const UI_SNAPSHOT_BYTESPACE: &str = "ui.snapshot.bytespace";
    /// Snapshot width in pixels for the presented surface.
    pub const UI_SNAPSHOT_WIDTH: &str = "ui.snapshot.width";
    /// Snapshot height in pixels for the presented surface.
    pub const UI_SNAPSHOT_HEIGHT: &str = "ui.snapshot.height";
    /// Snapshot stride in bytes per row (pixel surfaces).
    pub const UI_SNAPSHOT_STRIDE: &str = "ui.snapshot.stride";
    /// Snapshot format (e.g. RGBA8888) for pixel surfaces.
    pub const UI_SNAPSHOT_FORMAT: &str = "ui.snapshot.format";
    /// Monotonic present epoch for atomic snapshot presentation.
    ///
    /// Reserved for presenter-owned updates (Blossom).
    pub const UI_PRESENT_EPOCH: &str = "ui.present.epoch";
    /// Monotonic paint epoch (Bloom/Requester bumps, Blossom watches).
    pub const UI_PAINT_EPOCH: &str = "ui.paint.epoch";

    // Cursor Snapshot keys
    pub const UI_CURSOR_SNAPSHOT_BYTESPACE: &str = "ui.cursor.snapshot.bytespace";
    pub const UI_CURSOR_SNAPSHOT_WIDTH: &str = "ui.cursor.snapshot.width";
    pub const UI_CURSOR_SNAPSHOT_HEIGHT: &str = "ui.cursor.snapshot.height";
    pub const UI_CURSOR_SNAPSHOT_STRIDE: &str = "ui.cursor.snapshot.stride";
    pub const UI_CURSOR_SNAPSHOT_FORMAT: &str = "ui.cursor.snapshot.format";

    // Snapshot Semantics keys
    /// Snapshot semantic mode (0 = WRITE_ONCE, 1 = MUTABLE_DIRTY).
    /// Default is WRITE_ONCE if not specified.
    pub const UI_SNAPSHOT_MODE: &str = "ui.snapshot.mode";
    /// Dirty flag for MUTABLE_DIRTY mode (0 = clean, 1 = dirty).
    /// Compositor must not read snapshot while dirty=1.
    pub const UI_SNAPSHOT_DIRTY: &str = "ui.snapshot.dirty";
    /// Frozen flag indicating snapshot bytespace is immutable.
    /// Set automatically by presenter when committing in WRITE_ONCE mode.
    pub const UI_SNAPSHOT_FROZEN: &str = "ui.snapshot.frozen";

    /// Optional bytespace id for packed damage rects.
    ///
    /// Bloom may publish these as derived, non-authoritative hints. They must
    /// never be required for correctness.
    pub const UI_DAMAGE_RECTS_BYTESPACE: &str = "ui.damage.rects.bytespace";
    /// Tile asset bytespace id for UI_TILE nodes (e.g. SVG source).
    pub const UI_TILE_ASSET: &str = "ui.tile.asset";
    /// Optional tile state for UI_TILE nodes (0 = placeholder, 1 = ready).
    pub const UI_TILE_STATE: &str = "ui.tile.state";

    // Font Graph Properties
    pub const FONT_NAME: &str = "font.name";
    pub const FONT_STYLE: &str = "font.style";
    pub const FONT_FAMILY_KEY: &str = "font.family_key";
    pub const FONT_FACE_KEY: &str = "font.face_key";
    pub const FONT_WEIGHT: &str = "font.weight";
    pub const FONT_WIDTH: &str = "font.width";
    pub const FONT_SLOPE: &str = "font.slope";
    pub const FONT_BYTESPACE: &str = "font.bytespace";
    pub const FONT_SIZE_BYTES: &str = "font.size_bytes";
    pub const FONT_COVERAGE_RANGES: &str = "font.coverage_ranges";
    pub const FONT_COVERAGE_COUNT: &str = "font.coverage_count";
    pub const UI_FONT_STACK: &str = "ui.font_stack";
    pub const UI_FONT_DEBUG: &str = "ui.font_debug";

    // Font Glyph Properties
    pub const FONT_GLYPH_CODEPOINT: &str = "font.glyph.codepoint";
    pub const FONT_GLYPH_PX_SIZE: &str = "font.glyph.px_size";
    pub const FONT_GLYPH_RASTER_MODE: &str = "font.glyph.raster_mode";
    pub const FONT_GLYPH_ADVANCE: &str = "font.glyph.advance";
    pub const FONT_GLYPH_WIDTH: &str = "font.glyph.width";
    pub const FONT_GLYPH_HEIGHT: &str = "font.glyph.height";
    pub const FONT_GLYPH_OFFSET_X: &str = "font.glyph.offset_x";
    pub const FONT_GLYPH_OFFSET_Y: &str = "font.glyph.offset_y";
    pub const FONT_GLYPH_BITMAP: &str = "font.glyph.bitmap"; // Bytespace ID
    pub const FONT_GLYPH_CACHE_KEY: &str = "font.glyph.cache_key";

    // Font Request Properties
    pub const FONT_REQUEST_FACE: &str = "font.request.face";
    pub const FONT_REQUEST_CODEPOINT: &str = "font.request.codepoint";
    pub const FONT_REQUEST_PX_SIZE: &str = "font.request.px_size";
    pub const FONT_IMPORT_ASSET: &str = "font.import.asset";
    pub const FONT_IMPORT_STATUS: &str = "font.import.status";

    // Font Blob Properties (raw file backing)
    pub const FONT_BLOB_SHA256: &str = "font.blob.sha256";
    pub const FONT_BLOB_MIME: &str = "font.blob.mime";

    // Font Atlas Properties
    pub const FONT_ATLAS_BYTESPACE: &str = "font.atlas.bytespace";
    pub const FONT_ATLAS_WIDTH: &str = "font.atlas.width";
    pub const FONT_ATLAS_HEIGHT: &str = "font.atlas.height";
    pub const FONT_ATLAS_FORMAT: &str = "font.atlas.format"; // 0=A8, 1=RGBA8888
    pub const FONT_ATLAS_VERSION: &str = "font.atlas.version"; // Monotonic

    // SVG Cache Properties (Blossom service)
    pub const SVG_CONTENT_HASH: &str = "svg.content_hash";
    pub const SVG_VARIANT_HASH: &str = "svg.variant_hash";
    pub const SVG_RASTER_BYTESPACE: &str = "svg.raster.bytespace";
    pub const SVG_RASTER_WIDTH: &str = "svg.raster.width";
    pub const SVG_RASTER_HEIGHT: &str = "svg.raster.height";
    pub const SVG_RASTER_STRIDE: &str = "svg.raster.stride";
    pub const SVG_RASTER_FORMAT: &str = "svg.raster.format";

    // Asset System (unified)
    pub const ASSET_KIND: &str = "asset.kind";
    pub const ASSET_NAME: &str = "asset.name";
    pub const ASSET_SOURCE: &str = "asset.source";
    pub const ASSET_HASH: &str = "asset.hash";
    pub const ASSET_SIZE: &str = "asset.size";
    pub const ASSET_BYTESPACE: &str = "asset.bytespace";
    pub const ASSET_GENERATION: &str = "asset.generation";
    pub const ASSET_ERROR: &str = "asset.error";
    /// Boolean: 1 if asset successfully loaded and ready for use
    pub const ASSET_READY: &str = "asset.ready";
    /// ThingId of the XML_DOCUMENT node for parsed assets (SVG, XML, HTML)
    pub const ASSET_XML_DOCUMENT: &str = "asset.xml_document";

    // Content Provider System (unified content sources)
    /// Content source kind: "limine_module", "iso9660_disk", etc.
    pub const CONTENT_SOURCE_KIND: &str = "content.source.kind";
    /// Content source name (e.g., "boot", "cdrom0")
    pub const CONTENT_SOURCE_NAME: &str = "content.source.name";
    /// Content source priority for overlay resolution (higher wins)
    pub const CONTENT_SOURCE_PRIORITY: &str = "content.source.priority";
    /// Content source state: "ready", "error", "initializing"
    pub const CONTENT_SOURCE_STATE: &str = "content.source.state";
    /// Content source generation (bumps on refresh/remount)
    pub const CONTENT_SOURCE_GEN: &str = "content.source.gen";

    // File/Directory Properties
    /// File name (leaf name, not full path)
    pub const FILE_NAME: &str = "file.name";
    /// File size in bytes
    pub const FILE_SIZE: &str = "file.size";
    /// File content hash (SHA-256 first 8 bytes as u64)
    pub const FILE_HASH: &str = "file.hash";
    /// File MIME type (optional)
    pub const FILE_MIME: &str = "file.mime";
    /// File bytespace ID for content
    pub const FILE_BYTESPACE: &str = "file.bytespace";
    /// File source (ThingId of ContentSource)
    pub const FILE_SOURCE: &str = "file.source";
    /// Directory name (leaf name)
    pub const DIR_NAME: &str = "dir.name";
    /// Directory full path (optional, for quick lookups)
    pub const DIR_PATH: &str = "dir.path";

    // Service Contract Properties
    /// Service contract name (service canonical name)
    pub const SERVICE_CONTRACT_NAME: &str = "service.contract.name";
    /// Service contract watched kinds (bytespace containing list of kind names)
    pub const SERVICE_CONTRACT_WATCHED_KINDS: &str = "service.contract.watched_kinds";
    /// Service contract published kinds (bytespace containing list of kind names)
    pub const SERVICE_CONTRACT_PUBLISHED_KINDS: &str = "service.contract.published_kinds";
    /// Service contract published properties (bytespace containing list of property keys)
    pub const SERVICE_CONTRACT_PUBLISHED_PROPERTIES: &str = "service.contract.published_properties";
    /// Service contract idempotency flag (1 if idempotent, 0 if stateful)
    pub const SERVICE_CONTRACT_IDEMPOTENT: &str = "service.contract.idempotent";
    /// Service contract boot assumptions (bytespace, MUST be empty for watch-driven services)
    pub const SERVICE_CONTRACT_BOOT_ASSUMPTIONS: &str = "service.contract.boot_assumptions";
    /// Service contract status ("declared", "registered", "active", "error")
    pub const SERVICE_CONTRACT_STATUS: &str = "service.contract.status";
    /// Service contract version (monotonic, increments on contract updates)
    pub const SERVICE_CONTRACT_VERSION: &str = "service.contract.version";

    // Scheduler/Process Properties
    /// Task ID (u64)
    pub const PROC_TID: &str = "proc.tid";
    /// Task state as interned symbol (see `task_state` module for values)
    pub const PROC_STATE: &str = "proc.state";
    /// Task priority (0-4: Idle, Low, Normal, High, Realtime)
    pub const PROC_PRIORITY: &str = "proc.priority";
    /// Boolean: 1 for userspace thread, 0 for kernel
    pub const PROC_IS_USER: &str = "proc.is_user";
    /// Exit code when task dies (i32 stored as u64)
    pub const PROC_EXIT_CODE: &str = "proc.exit_code";
    /// Interned string name of the task/module
    pub const PROC_NAME: &str = "proc.name";
    /// Initial startup argument passed via `spawn_with_arg` (usize stored as u64).
    /// Set on the spawned thread node so supervisors can observe it without
    /// out-of-band channels.
    pub const PROC_SPAWN_ARG: &str = "proc.spawn_arg";

    // Launch/Event Properties
    /// Monotonic timestamp for last launch (nanoseconds since boot)
    pub const LAUNCH_AT: &str = "launch.at";
}

/// String values used for the `proc.state` property on task nodes.
///
/// These are interned symbols stored as `u64` in the graph; compare the
/// `proc.state` property value against `sys::intern(task_state::RUNNABLE)`, etc.
pub mod task_state {
    /// Thread is ready to run (not yet scheduled onto a CPU).
    pub const RUNNABLE: &str = "runnable";
    /// Thread is currently executing on a CPU.
    pub const RUNNING: &str = "running";
    /// Thread is blocked waiting for a resource (port, lock, etc.).
    pub const BLOCKED: &str = "blocked";
    /// Thread is sleeping until a deadline.
    pub const SLEEPING: &str = "sleeping";
    /// Thread has exited; `proc.exit_code` holds the final exit code.
    pub const DEAD: &str = "dead";
}

pub mod kinds {
    pub const DEV_HOST: &str = "dev.Host";
    pub const DEV_BUS_PLATFORM: &str = "dev.bus.Platform";
    pub const FW_TABLE_ACPI: &str = "fw.table.Acpi";
    pub const FW_TABLE_DTB: &str = "fw.table.Dtb";
    pub const MEM_RANGE: &str = "mem.Range";
    pub const DEV_RTC_CMOS: &str = "dev.rtc.Cmos";
    pub const MEMFD: &str = "MemFd";
    pub const BYTESPACE: &str = "Bytespace"; // Legacy alias
    pub const RES_IO_PORT_RANGE: &str = "res.io.PortRange";
    pub const DEV_DISPLAY_FRAMEBUFFER: &str = "dev.display.Framebuffer";
    pub const PROC_KERNEL: &str = "proc.Kernel";
    pub const SVC_ROOT: &str = "svc.Root";
    pub const BOOT_MODULE: &str = "boot.Module";
    pub const DEV_CPU: &str = "dev.Cpu";
    pub const SVC_SCHEDULER: &str = "svc.Scheduler";
    pub const PROC_TASK: &str = "proc.Task";
    pub const PROC_THREAD: &str = "proc.Thread";
    pub const MEM_PAGE: &str = "mem.Page";
    pub const MEM_STACK: &str = "mem.Stack";
    pub const MEM_HEAP: &str = "mem.Heap";

    // Render artifact kinds (renderer-owned derived nodes)
    pub const RENDER_CACHE_ROOT: &str = "render.CacheRoot";
    pub const RENDER_PATH: &str = "render.Path";
    pub const RENDER_FLATTENED_PATH: &str = "render.FlattenedPath";
    pub const RENDER_EDGE_LIST: &str = "render.EdgeList";
    pub const RENDER_SCANLINE_SPANS: &str = "render.ScanlineSpans";
    pub const RENDER_COVERAGE_TILE: &str = "render.CoverageTile";
    pub const RENDER_PAINT: &str = "render.Paint";
    pub const RENDER_RASTER_TILE: &str = "render.RasterTile";
    pub const RENDER_SNAPSHOT: &str = "render.Snapshot";
    pub const RENDER_DEBUG_STATS: &str = "render.Debug.Stats";
    pub const FW_BOOT: &str = "fw.Boot";
    pub const TIME_WALL_CLOCK_SAMPLE: &str = "time.WallClockSample";
    pub const SVC_TIME_SYSTEM_CLOCK: &str = "svc.time.SystemClock";
    pub const DEV_BUS_PCI: &str = "dev.bus.Pci";
    pub const DEV_PCI_FUNCTION: &str = "dev.pci.Function";
    // LPC / Legacy IO
    pub const DEV_BRIDGE_LPC: &str = "dev.bridge.Lpc";
    pub const DEV_BUS_LEGACY_IO: &str = "dev.bus.LegacyIo";
    pub const DEV_INPUT_PS2_CONTROLLER: &str = "dev.input.Ps2Controller";
    pub const CAP_IOPORT_RANGE: &str = "cap.ioport.Range";
    // Virtio GPU
    pub const DEV_DISPLAY_GPU: &str = "dev.display.Gpu";
    pub const DEV_DISPLAY_SCANOUT: &str = "dev.display.Scanout";
    // Network
    pub const DEV_NET_NIC: &str = "dev.net.Nic";
    pub const DEV_NET_PCI_STUB: &str = "dev.net.PciStub";
    pub const DEV_NET_WLAN_PCI_STUB: &str = "dev.net.WlanPciStub";
    pub const NET_HOST: &str = "net.Host";
    pub const NET_ADDRESS: &str = "net.Address";
    pub const NET_SERVICE_TYPE: &str = "net.ServiceType";
    pub const NET_SERVICE_INSTANCE: &str = "net.ServiceInstance";
    pub const NET_ENDPOINT: &str = "net.Endpoint";
    pub const NET_SOCKET: &str = "net.Socket";
    pub const NET_CONNECTION: &str = "net.Connection";
    pub const NET_TXT_RECORD: &str = "net.TxtRecord";
    pub const NET_ADVERTISE_ROOT: &str = "net.AdvertiseRoot";
    pub const NET_SERVICE_INSTANCE_DESIRED: &str = "net.ServiceInstanceDesired";
    // Storage
    pub const DEV_STORAGE_DISK: &str = "dev.storage.Disk";
    pub const DEV_STORAGE_PARTITION: &str = "dev.storage.Partition";
    pub const DEV_STORAGE_BLOCK_DEVICE: &str = "dev.storage.BlockDevice";
    pub const SVC_STORAGE: &str = "svc.Storage";
    pub const LOG_ENTRY: &str = "log.Entry";

    // Audio
    pub const DEV_SOUND: &str = "dev.sound.Virtio";
    pub const DEV_SOUND_HDA_PCI_STUB: &str = "dev.sound.HdaPciStub";

    // USB
    pub const DEV_USB_XHCI_PCI_STUB: &str = "dev.usb.XhciPciStub";

    // Display (non-virtio PCI fallback)
    pub const DEV_DISPLAY_GPU_PCI_STUB: &str = "dev.display.GpuPciStub";

    // UI Kinds
    pub const UI_CROWN: &str = "ui.Crown";
    pub const UI_WINDOW: &str = "ui.Window";
    pub const UI_PANEL: &str = "ui.Panel";
    pub const UI_TEXT: &str = "ui.Text";
    pub const UI_IMAGE: &str = "ui.Image";
    pub const UI_OVERLAY: &str = "ui.Overlay";
    pub const UI_INLINE: &str = "ui.Inline";
    pub const UI_VIEWPORT: &str = "ui.Viewport";
    pub const UI_TILE: &str = "ui.Tile";
    pub const UI_TEXT_RUN: &str = "ui.TextRun";
    pub const UI_CHROME: &str = "ui.Chrome";
    pub const UI_NODE: &str = "ui.Node";
    pub const UI_BUTTON: &str = "ui.Button";
    pub const UI_CHECKBOX: &str = "ui.Checkbox";
    pub const UI_COLUMN: &str = "ui.Container.Column";
    pub const UI_LIST_ITEM: &str = "ui.ListItem";
    /// Graph-native drawlist (stable identity, packed ops in bytespace).
    pub const UI_DRAWLIST: &str = "ui.DrawList";

    // Font Graph Kinds
    pub const FONT_SUPERFAMILY: &str = "font.Superfamily";
    pub const FONT_FAMILY: &str = "font.Family";
    pub const FONT_FACE: &str = "font.Face";
    pub const FONT_FILE: &str = "font.File";
    pub const FONT_COVERAGE: &str = "font.Coverage";
    pub const FONT_INSTANCE: &str = "font.Instance";
    pub const FONT_GLYPH: &str = "font.Glyph";
    pub const FONT_BLOB: &str = "font.Blob"; // Raw font file backing store
    pub const FONT_ATLAS: &str = "font.Atlas"; // Glyph atlas for (face, size)
    pub const FONT_IMPORT_REQUEST: &str = "font.ImportRequest";
    pub const FONT_GLYPH_REQUEST: &str = "font.GlyphRequest";

    pub const CLOCK: &str = "Clock";

    pub const BINDING: &str = "Binding";

    // XML Graph Kinds
    pub const XML_DOCUMENT: &str = "xml.Document";
    pub const XML_ELEMENT: &str = "xml.Element";
    pub const XML_ATTRIBUTE: &str = "xml.Attribute";
    pub const XML_TEXT: &str = "xml.Text";

    // HTML Graph Kinds
    pub const HTML_DOCUMENT: &str = "html.Document";
    pub const HTML_ELEMENT: &str = "html.Element";
    pub const HTML_ATTRIBUTE: &str = "html.Attribute";
    pub const HTML_TEXT: &str = "html.Text";
    pub const HTML_COMMENT: &str = "html.Comment";

    // CSS Graph Kinds
    pub const CSS_STYLESHEET: &str = "css.Stylesheet";
    pub const CSS_RULE: &str = "css.Rule";
    pub const CSS_SELECTOR: &str = "css.Selector";
    pub const CSS_DECLARATION: &str = "css.Declaration";
    pub const CSS_AT_RULE: &str = "css.AtRule";

    // SVG Cache Kinds (Blossom service)
    pub const SVG_ASSET: &str = "svg.Asset";
    pub const SVG_RASTER_VARIANT: &str = "svg.RasterVariant";

    pub const SVC_INIT: &str = "svc.Init";
    pub const SVC_CAMBIUM: &str = "svc.Cambium";
    pub const TIME_CLOCK: &str = "time.Clock";
    pub const TIME_TIMER: &str = "time.Timer";
    pub const UI_THEME: &str = "ui.Theme";
    pub const UI_WIDGET: &str = "ui.Widget";

    pub const ASSET: &str = "Asset";
    pub const ASSET_REQUEST: &str = "AssetRequest";

    // Content Provider System
    /// A content source that provides files/directories (Limine modules, ISO, etc.)
    pub const CONTENT_SOURCE: &str = "content.Source";
    /// A directory in the filesystem graph
    pub const CONTENT_DIR: &str = "fs.Directory";
    /// A file in the filesystem graph
    pub const CONTENT_FILE: &str = "fs.File";

    // Service Contract Kinds
    /// Service contract node (registered at /sys/services/{name})
    pub const SERVICE_CONTRACT: &str = "svc.Contract";
    /// Service instance node (running service)
    pub const SERVICE_INSTANCE: &str = "svc.Instance";
}

/// Snapshot semantics and constants for UI presentation surfaces.
///
/// # The Snapshot Contract
///
/// ## Write-Once Mode (Default)
///
/// 1. Painter allocates a bytespace and renders into it.
/// 2. Painter sets UI_SNAPSHOT_* metadata (bytespace, width, height, stride, format).
/// 3. Painter sets UI_SNAPSHOT_FROZEN=1 (optional explicit freeze).
/// 4. Painter sets UI_PRESENT_EPOCH to commit the snapshot.
/// 5. Compositor reads snapshot; bytespace must not be mutated.
/// 6. For next frame, painter allocates NEW bytespace and repeats.
///
/// ## Invariants
///
/// - A snapshot bytespace with a non-zero epoch MUST NOT be mutated.
/// - If UI_SNAPSHOT_MODE is unset, assume WRITE_ONCE.
/// - Two successive frames MUST NOT alias the same bytespace unless
///   the first frame's epoch has been superseded.
///
/// ## Anti-Aliasing Guarantee
///
/// ```text
/// Frame N:   epoch=5, bytespace=0x1234
/// Frame N+1: epoch=6, bytespace=0x5678  // MUST be different
///                                        // OR epoch=5 still present (no update)
/// ```
pub mod ui_snapshot {
    /// Pixel format for RGBA8888 surfaces.
    #[deprecated(note = "Use pixel_format::BGRA8888 instead")]
    pub const PIXEL_FORMAT_RGBA8888: u64 = 1;
}

/// Snapshot semantic modes for UI presentation surfaces.
pub mod snapshot_mode {
    /// Write-once mode: snapshot is immutable after presentation.
    /// The bytespace becomes frozen when UI_PRESENT_EPOCH is set.
    /// Updates require creating a new bytespace and atomically
    /// replacing UI_SNAPSHOT_BYTESPACE.
    pub const WRITE_ONCE: u64 = 0;

    /// Mutable mode with dirty tracking (reserved for future use).
    /// The bytespace can be modified in place; writers must set
    /// UI_SNAPSHOT_DIRTY=1 before mutation and clear it after.
    /// Compositor must check dirty flag and skip/retry if set.
    pub const MUTABLE_DIRTY: u64 = 1;
}

/// Canonical pixel format constants matching `abi::pixel::PixelFormat`.
pub mod pixel_format {
    /// Unknown or unsupported format.
    pub const UNKNOWN: u64 = 0;
    /// 32-bit BGRA: Memory [B, G, R, A] -> u32 0xAARRGGBB.
    pub const BGRA8888: u64 = 1;
    /// 32-bit BGRX: Memory [B, G, R, X] -> u32 0xXXRRGGBB (alpha ignored).
    pub const BGRX8888: u64 = 2;
    /// 16-bit RGB565: No alpha channel.
    pub const RGB565: u64 = 3;
}

/// UI kind tags stored in `keys::UI_KIND`.
pub mod ui_kind {
    pub const BUTTON: u64 = 1;
    pub const CHECKBOX: u64 = 2;
    pub const TEXT: u64 = 3;
    pub const COLUMN: u64 = 4;
    pub const WINDOW: u64 = 5;
    pub const ROW: u64 = 6;
    pub const TEXT_INPUT: u64 = 7;
    pub const SPACER: u64 = 8;
    pub const SEPARATOR: u64 = 9;
    pub const LIST_ITEM: u64 = 10;
}

pub mod rels {
    pub const HAS_BUS: &str = "HAS_BUS";
    pub const HAS_DEVICE: &str = "HAS_DEVICE";
    pub const HAS_RESOURCE: &str = "HAS_RESOURCE";
    pub const HAS_FIRMWARE: &str = "HAS_FIRMWARE";
    pub const PROVIDES_TABLE: &str = "PROVIDES_TABLE";
    pub const BACKED_BY: &str = "BACKED_BY";
    pub const DERIVED_FROM: &str = "DERIVED_FROM";
    pub const RUNS_ON: &str = "RUNS_ON";
    pub const PROVIDES: &str = "PROVIDES";
    pub const HAS_CPU: &str = "HAS_CPU";
    pub const HAS_MEMORY_RANGE: &str = "HAS_MEMORY_RANGE";
    pub const HAS_MODULE: &str = "HAS_MODULE";
    pub const HAS_SERVICE: &str = "HAS_SERVICE";
    pub const MONITORS: &str = "MONITORS";
    pub const SEEDED_BY: &str = "SEEDED_BY";
    pub const USES: &str = "USES";
    // LPC / Legacy IO
    pub const IMPLEMENTS: &str = "IMPLEMENTS";
    pub const USES_IOPORTS: &str = "USES_IOPORTS";
    pub const HAS_SCANOUT: &str = "HAS_SCANOUT";

    // UI Relations
    pub const CHILD_OF: &str = "CHILD_OF";
    pub const HAS_CHILD: &str = "HAS_CHILD";
    pub const CLIP_TO: &str = "CLIP_TO";
    pub const ROOT_UI: &str = "ROOT_UI";

    // Font Graph Relationships
    pub const FONT_CONTAINS: &str = "font.contains";
    pub const FONT_COVERS: &str = "font.covers";
    pub const FONT_MEMBER_OF: &str = "font.member_of";
    pub const FONT_FALLBACK_AFTER: &str = "font.fallback_after";
    pub const FONT_ALIAS: &str = "font.alias";
    pub const FONT_HAS_FACE: &str = "font.has_face";
    pub const FONT_HAS_ASSET: &str = "font.has_asset";
    pub const FONT_HAS_COVERAGE: &str = "font.has_coverage";
    pub const FONT_HAS_GLYPH: &str = "font.has_glyph";
    pub const FONT_HAS_INSTANCE: &str = "font.has_instance";
    pub const FONT_HAS_RESULT: &str = "font.has_result";
    pub const FONT_HAS_BLOB: &str = "font.has_blob"; // Face -> Blob (raw file)
    pub const FONT_HAS_ATLAS: &str = "font.has_atlas"; // Face -> Atlas
    pub const FONT_FALLBACK_TO: &str = "font.fallback_to";

    // XML Relationships
    pub const HAS_ROOT: &str = "HAS_ROOT";
    pub const HAS_ATTR: &str = "HAS_ATTR";

    // SVG Cache Relationships (Blossom service)
    pub const SVG_VARIANT_OF: &str = "svg.variant_of";
    pub const SVG_HAS_PIXELS: &str = "svg.has_pixels";

    // Content Provider Relationships
    /// Content source contains a directory or file
    pub const CONTENT_CONTAINS: &str = "content.contains";
    /// File/directory located at parent directory
    pub const CONTENT_LOCATED_AT: &str = "content.located_at";
    /// Content provided by source
    pub const CONTENT_PROVIDED_BY: &str = "content.provided_by";

    // Service Contract Relationships
    /// Service implements contract (Service -> Contract)
    pub const IMPLEMENTS_CONTRACT: &str = "IMPLEMENTS_CONTRACT";
    /// Service requires another service (Service -> Service)
    pub const REQUIRES_SERVICE: &str = "REQUIRES_SERVICE";
    /// Service watches a node kind (Contract -> Kind)
    pub const WATCHES_KIND: &str = "WATCHES_KIND";
    /// Service publishes a node kind (Contract -> Kind)
    pub const PUBLISHES_KIND: &str = "PUBLISHES_KIND";

    // Scheduler/Process Relationships
    /// Scheduler owns/manages a task (svc.Scheduler -> proc.Thread)
    pub const SCHED_HAS_TASK: &str = "SCHED_HAS_TASK";
    /// Task is child of another task (proc.Thread -> proc.Thread)
    pub const TASK_CHILD_OF: &str = "TASK_CHILD_OF";
    /// Task was spawned by another task (semantic "caused" edge)
    pub const TASK_SPAWNED: &str = "TASK_SPAWNED";
    /// Thread uses a bytespace mapping (proc.Thread -> Bytespace)
    pub const THREAD_USES_BYTESPACE: &str = "THREAD_USES_BYTESPACE";
    /// Task is pinned to a specific CPU (proc.Thread -> dev.Cpu)
    pub const PINNED_TO: &str = "PINNED_TO";

    /// Host launched a module (dev.Host -> boot.Module)
    pub const LAUNCHED: &str = "LAUNCHED";
    pub const NET_HOST_HAS_ADDR: &str = "net.host_has_addr";
    pub const NET_HOST_ADVERTISES: &str = "net.host_advertises";
    pub const NET_INSTANCE_IS_A: &str = "net.instance_is_a";
    pub const NET_INSTANCE_REACHABLE_AT: &str = "net.instance_reachable_at";
    pub const NET_INSTANCE_HAS_TXT: &str = "net.instance_has_txt";
    pub const NET_WANTS_ADVERTISED: &str = "net.wants_advertised";
    pub const PROC_OWNS_SOCKET: &str = "proc.owns_socket";
    pub const NET_SOCKET_HAS_LOCAL: &str = "net.socket_has_local";
    pub const NET_SOCKET_HAS_REMOTE: &str = "net.socket_has_remote";
    pub const NET_SOCKET_HAS_CONNECTION: &str = "net.socket_has_connection";
    pub const NET_CONNECTION_PEER: &str = "net.connection_peer";
    pub const NET_CONNECTION_FOR_SERVICE: &str = "net.connection_for_service";
}

// Virtio GPU additions
pub mod virtio {
    pub const VENDOR_ID: u16 = 0x1af4;
    pub const GPU_DEVICE_ID: u16 = 0x1050;
}

// HID / Input additions (Bristle v0)
pub mod hid {
    // Service kinds
    pub const SVC_INPUT: &str = "svc.Input"; // Bristle broker

    // Device kinds
    pub const DEV_HID_KEYBOARD: &str = "dev.hid.Keyboard";
    pub const DEV_HID_MOUSE: &str = "dev.hid.Mouse";
    pub const DEV_HID_TOUCHPAD: &str = "dev.hid.Touchpad";
    pub const DEV_HID_GAMEPAD: &str = "dev.hid.Gamepad";

    // Driver kinds
    pub const DRV_PS2_KEYBOARD: &str = "drv.Ps2Keyboard";
    pub const DRV_PS2_MOUSE: &str = "drv.Ps2Mouse";

    // Relations
    pub const REL_CONSUMES: &str = "CONSUMES"; // (svc.Input)-[:CONSUMES]->(dev.hid.Keyboard)
    pub const REL_ROUTES_TO: &str = "ROUTES_TO"; // (svc.Input)-[:ROUTES_TO]->(app.Echo)
    pub const REL_PRODUCES: &str = "PRODUCES"; // (drv.Ps2Keyboard)-[:PRODUCES]->(dev.hid.Keyboard)
}

// Pointer / Input properties
pub mod pointer {
    pub const POINTER_X: &str = "pointer.x";
    pub const POINTER_Y: &str = "pointer.y";
    pub const POINTER_BUTTONS: &str = "pointer.buttons";
    pub const POINTER: &str = "input.Pointer";
}

// Keyboard / Input properties
pub mod keyboard {
    /// Current modifier bitset (Mods.0 value)
    pub const KEYBOARD_MODS: &str = "keyboard.mods";
    /// Last key code (Key as u16)
    pub const KEYBOARD_LAST_KEY: &str = "keyboard.last_key";
    /// Last key edge: 0=up, 1=down
    pub const KEYBOARD_KEY_EDGE: &str = "keyboard.key_edge";
    /// Monotonic generation for key events (increments on each key event)
    pub const KEYBOARD_GEN: &str = "keyboard.gen";
}
