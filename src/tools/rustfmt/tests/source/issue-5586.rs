// rustfmt-version: Two
fn main() {
    // sample 1
    {
        {
            {
                {
                    {
                        let push_ident =
                            if let Some(&node_id) = subgraph_nodes.get(pull_to_push_idx) {
                                self.node_id_as_ident(node_id, false)
                            } else {
                                // Entire subgraph is pull (except for a single send/push handoff output).
                                assert_eq!(
    1,
    send_ports.len(),
    "If entire subgraph is pull, should have only one handoff output."
);
                                send_ports[0].clone()
                            };
                    }
                }
            }
        }
    }

    // sample 2
    {
        {
            {
                {
                    {
                        let push_ident = if let Some(&node_id) =
                            subgraph_nodes.get(pull_to_push_idx)
                        {
                            self.node_id_as_ident(node_id, false)
                        } else {
                            // Entire subgraph is pull (except for a single send/push handoff output).
                            assert_eq!(
                                1,
                                send_ports.len(),
                                "If entire subgraph is pull, should have only one handoff output."
                            );
                            send_ports[0].clone()
                        };
                    }
                }
            }
        }
    }

    // sample 3
    {
        {
            {
                {
                    {
                        let push_ident =
                            if let Some(&node_id) = subgraph_nodes.get(pull_to_push_idx) {
                                self.node_id_as_ident(node_id, false)
                            } else {
                                // Entire subgraph is pull (except for a single send/push handoff output).
                                assert_eq!(
                    1,
                    send_ports.len(),
                    "If entire subgraph is pull, should have only one handoff output."
                );
                                send_ports[0].clone()
                            };
                    }
                }
            }
        }
    }

    // sample 4
    {{{{{
        let push_ident =
            if let Some(&node_id) = subgraph_nodes.get(pull_to_push_idx) {
                self.node_id_as_ident(node_id, false)
            } else {
                // Entire subgraph is pull (except for a single send/push handoff output).
                assert_eq!(
                    1,
                    send_ports.len(),
                    "If entire subgraph is pull, should have only one handoff output."
                );
                send_ports[0].clone()
            };
    }}}}}

    // sample 5
    {
        {
            {
                {
                    {
                        let push_ident =
                            if let Some(&node_id) = subgraph_nodes.get(pull_to_push_idx) {
                                self.node_id_as_ident(node_id, false)
                            } else {
                                // Entire subgraph is pull (except for a single send/push handoff output).
                                assert_eq!(
                                1,
                                send_ports.len(),
                                "If entire subgraph is pull, should have only one handoff output."
                            );
                                send_ports[0].clone()
                            };
                    }
                }
            }
        }
    }
    
    // sample 6
    {
        {
            {
                {
                    {
                        let push_ident = if let Some(&node_id) =
                            subgraph_nodes.get(pull_to_push_idx)
                        {
                            self.node_id_as_ident(node_id, false)
                        } else {
                            // Entire subgraph is pull (except for a single send/push handoff output).
                            assert_eq!(
                                1,
                                send_ports.len(),
                                "If entire subgraph is pull, should have only one handoff output."
                            );
                            send_ports[0].clone()
                        };
                    }
                }
            }
        }
    }

    // sample 7
    {
        {
            {
                {
                    {
                        let push_ident =
                            if let Some(&node_id) = subgraph_nodes.get(pull_to_push_idx) {
                                self.node_id_as_ident(node_id, false)
                            } else {
                                // Entire subgraph is pull (except for a single send/push handoff output).
                                assert_eq!(
                                    1,
                                    send_ports.len(),
                                    "If entire subgraph is pull, should have only one handoff output."
                                );
                                send_ports[0].clone()
                            };
                    }
                }
            }
        }
    }
}
